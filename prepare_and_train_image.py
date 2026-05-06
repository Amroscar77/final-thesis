"""
Prepare Image Dataset + Train Image Model
==========================================
Step 1: Scan uploaded_videos/, run the existing video model on each video,
        extract face-cropped frames, save them labelled into training_data/real/ & fake/
Step 2: Train the ResNeXt image classifier on those frames.

Run:
    python prepare_and_train_image.py
"""

import os, sys, glob, time, random, gc, shutil
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(__file__)
APP_DIR = os.path.join(BASE,
    "Deepfake_detection_using_deep_learning", "Django Application")
VIDEO_DIR  = os.path.join(APP_DIR, "uploaded_videos")
MODELS_DIR = os.path.join(APP_DIR, "models")
DATA_DIR   = os.path.join(BASE, "training_data")
REAL_DIR   = os.path.join(DATA_DIR, "real")
FAKE_DIR   = os.path.join(DATA_DIR, "fake")

# ── config ───────────────────────────────────────────────────────────────────
SEQUENCE_LEN     = 20     # frames fed to video model for labelling
FRAMES_PER_VIDEO = 15     # face frames saved per video for training
MIN_CONFIDENCE   = 70.0   # only accept predictions above this confidence
IMG_SIZE         = 224
EPOCHS           = 20
BATCH_SIZE       = 8
LR               = 1e-4
VAL_SPLIT        = 0.2

device = 'cpu'  # no GPU on this machine

# ── normalisation (ImageNet) ──────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

video_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
sm = nn.Softmax(dim=1)

# =============================================================================
# 1.  VIDEO MODEL  (existing ResNeXt-50 + LSTM — V1)
# =============================================================================
class VideoModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnext50_32x4d(pretrained=False)
        self.model   = nn.Sequential(*list(base.children())[:-2])
        self.lstm    = nn.LSTM(2048, 2048, 1, False)  # bias=False matches saved checkpoints
        self.relu    = nn.LeakyReLU()
        self.dp      = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x    = x.view(B * T, C, H, W)
        fmap = self.model(x)
        x    = self.avgpool(fmap)
        x    = x.view(B, T, 2048)
        x, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x[:, -1, :]))


def pick_best_video_model():
    """Return path of the best available 20-frame V1 model."""
    candidates = glob.glob(os.path.join(MODELS_DIR, "*20*frames*.pt"))
    if not candidates:
        candidates = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
        candidates = [m for m in candidates if "image" not in m and "v2" not in m]
    if not candidates:
        return None
    # pick highest accuracy (encoded in filename position [1])
    def acc(p):
        try: return int(os.path.basename(p).split("_")[1])
        except: return 0
    return max(candidates, key=acc)


def load_video_model(path):
    m = VideoModel(2)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    m.eval()
    return m


def predict_video(model, frames_tensor):
    """frames_tensor: (1, T, 3, 112, 112). Returns (label, confidence)."""
    with torch.no_grad():
        _, logits = model(frames_tensor)
        probs = sm(logits)
        pred  = int(torch.argmax(probs, 1).item())
        conf  = float(probs[0, pred].item()) * 100
    return pred, conf   # pred: 1=REAL, 0=FAKE


# =============================================================================
# 2.  FACE DETECTION
# =============================================================================
_cascade = None

def get_face(frame_bgr, padding=20):
    """Crop face from BGR frame. Returns RGB crop or full frame."""
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        H, W = rgb.shape[:2]
        x1 = max(0, x - padding);     y1 = max(0, y - padding)
        x2 = min(W, x+w+padding);     y2 = min(H, y+h+padding)
        crop = rgb[y1:y2, x1:x2]
        if crop.size > 0:
            return crop
    return rgb


# =============================================================================
# 3.  DATASET BUILDER
# =============================================================================
def extract_frames_from_video(video_path, n_frames=20):
    """Return n_frames evenly-spaced BGR frames."""
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    n      = min(n_frames, total)
    idxs   = [int(i) for i in np.linspace(0, total - 1, n)]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    return frames


def frames_to_tensor(frames):
    """Stack BGR frames into (1, T, 3, 112, 112) tensor for video model."""
    tensors = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        tensors.append(video_tf(rgb))
    if not tensors:
        return None
    t = torch.stack(tensors).unsqueeze(0)   # (1, T, 3, 112, 112)
    return t


def build_dataset(video_model):
    """
    For every video in uploaded_videos/:
      - predict label + confidence with video model
      - if confidence >= MIN_CONFIDENCE, extract face frames
      - save to training_data/real/ or training_data/fake/
    """
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    videos = [v for v in glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
              if os.path.getsize(v) > 10_000]   # skip tiny files

    if not videos:
        print("[ERROR] No videos found in uploaded_videos/")
        return 0, 0

    n_real = n_fake = 0
    print(f"\n[STEP 1] Labelling {len(videos)} videos with the video model...")
    print(f"         Minimum confidence to accept: {MIN_CONFIDENCE}%\n")

    for i, vpath in enumerate(videos, 1):
        vname = os.path.basename(vpath)
        frames = extract_frames_from_video(vpath, n_frames=SEQUENCE_LEN)
        if len(frames) < 5:
            print(f"  [{i:02d}/{len(videos)}] SKIP  {vname}  (too short)")
            continue

        tensor = frames_to_tensor(frames)
        if tensor is None:
            continue

        label, conf = predict_video(video_model, tensor)
        del tensor; gc.collect()

        if conf < MIN_CONFIDENCE:
            print(f"  [{i:02d}/{len(videos)}] SKIP  {vname}  "
                  f"({'REAL' if label==1 else 'FAKE'} {conf:.1f}% — low confidence)")
            continue

        # Extract face frames for image training
        label_str = "REAL" if label == 1 else "FAKE"
        out_dir   = REAL_DIR if label == 1 else FAKE_DIR
        saved     = 0

        save_frames = extract_frames_from_video(vpath, n_frames=FRAMES_PER_VIDEO)
        for j, frame_bgr in enumerate(save_frames):
            face = get_face(frame_bgr)
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            ts   = int(time.time() * 1000)
            fname = f"{os.path.splitext(vname)[0]}_f{j}_{ts}.png"
            cv2.imwrite(os.path.join(out_dir, fname), face_bgr)
            saved += 1

        if label == 1:
            n_real += saved
        else:
            n_fake += saved

        print(f"  [{i:02d}/{len(videos)}] {label_str:4s} {conf:5.1f}%  "
              f"{vname}  -> saved {saved} frames")

    print(f"\n[STEP 1 DONE]  Real frames: {n_real}  |  Fake frames: {n_fake}")
    return n_real, n_fake


# =============================================================================
# 4.  IMAGE MODEL  (ResNeXt-50 without LSTM)
# =============================================================================
class ImageModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5, freeze_backbone=False):
        super().__init__()
        backbone = models.resnext50_32x4d(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =============================================================================
# 5.  IMAGE DATASET
# =============================================================================
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_dataset_folder(root):
    paths, labels = [], []
    for lname, lval in [('real', 1), ('fake', 0)]:
        folder = os.path.join(root, lname)
        for ext in IMG_EXTS:
            for p in glob.glob(os.path.join(folder, f'*{ext}')) + \
                     glob.glob(os.path.join(folder, f'*{ext.upper()}')):
                paths.append(p); labels.append(lval)
    return paths, labels


# =============================================================================
# 6.  TRAINING UTILITIES
# =============================================================================
class Meter:
    def __init__(self): self.reset()
    def reset(self): self.s = self.n = 0.0
    def update(self, v, n=1): self.s += v*n; self.n += n
    @property
    def avg(self): return self.s / max(self.n, 1)

def accuracy(out, tgt):
    _, p = out.topk(1, 1)
    return 100.0 * p.t().eq(tgt.view(1,-1)).float().sum().item() / tgt.size(0)

def run_train(model, loader, crit, opt):
    model.train()
    L, A = Meter(), Meter()
    for x, y in loader:
        x, y = x.to(device), y.to(device).long()
        opt.zero_grad()
        out  = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        L.update(loss.item(), x.size(0))
        A.update(accuracy(out, y), x.size(0))
    return L.avg, A.avg

def run_eval(model, loader, crit):
    model.eval()
    L, A = Meter(), Meter()
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).long()
            out  = model(x)
            loss = crit(out, y)
            L.update(loss.item(), x.size(0))
            A.update(accuracy(out, y), x.size(0))
            _, p = torch.max(out, 1)
            all_p.extend(p.cpu().tolist())
            all_t.extend(y.cpu().tolist())
    return L.avg, A.avg, all_p, all_t

def save_curves(tl, vl, ta, va, out_dir):
    e = range(1, len(tl)+1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(e, tl, 'g-o', label='Train Loss')
    a1.plot(e, vl, 'b-o', label='Val Loss')
    a1.set_title('Loss'); a1.set_xlabel('Epoch')
    a1.legend(); a1.grid(alpha=.3)
    a2.plot(e, ta, 'g-o', label='Train Acc')
    a2.plot(e, va, 'b-o', label='Val Acc')
    a2.set_title('Accuracy'); a2.set_xlabel('Epoch')
    a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout()
    out = os.path.join(out_dir, 'image_training_curves.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"[INFO] Curves saved -> {out}")


# =============================================================================
# 7.  MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  DeepScan — Image Model Training Pipeline")
    print("=" * 60)

    # ── Step 1: load video teacher model ─────────────────────────
    model_path = pick_best_video_model()
    if not model_path:
        print("[ERROR] No trained video model found in models/")
        sys.exit(1)
    print(f"\n[INFO] Teacher model : {os.path.basename(model_path)}")
    print("[INFO] Loading teacher model (may take ~30 seconds)...")
    video_model = load_video_model(model_path)
    print("[INFO] Teacher model loaded.")

    # ── Step 2: build image dataset ───────────────────────────────
    n_real, n_fake = build_dataset(video_model)
    del video_model; gc.collect()

    total = n_real + n_fake
    if total < 10:
        print(f"\n[ERROR] Only {total} images collected — need at least 10.")
        print("        Upload more videos via the web app, then re-run.")
        sys.exit(1)

    print(f"\n[INFO] Dataset ready: {n_real} real frames, {n_fake} fake frames")
    print(f"[INFO] Saved to: {DATA_DIR}")

    # ── Step 3: load all image paths ──────────────────────────────
    paths, labels = load_dataset_folder(DATA_DIR)
    print(f"[INFO] Total images for training: {len(paths)}")

    combined = list(zip(paths, labels))
    random.shuffle(combined)
    paths   = [x[0] for x in combined]
    labels  = [x[1] for x in combined]

    split       = int(len(paths) * (1 - VAL_SPLIT))
    tr_p, tr_l  = paths[:split], labels[:split]
    vl_p, vl_l  = paths[split:], labels[split:]
    print(f"[INFO] Train: {len(tr_p)}  |  Val: {len(vl_p)}")

    # ── Step 4: transforms ────────────────────────────────────────
    norm = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.1),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm,
    ])

    tr_ds = ImageDataset(tr_p, tr_l, transform=train_tf)
    vl_ds = ImageDataset(vl_p, vl_l, transform=val_tf)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Step 5: build image model ─────────────────────────────────
    print(f"\n[STEP 2] Training Image Model on {device.upper()}")
    print(f"         Architecture : ResNeXt-50 (no LSTM)")
    print(f"         Epochs       : {EPOCHS}")
    print(f"         Batch size   : {BATCH_SIZE}")
    print(f"         Image size   : {IMG_SIZE}x{IMG_SIZE}\n")

    model = ImageModel(num_classes=2, freeze_backbone=False).to(device)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_p:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # ── Step 6: training loop ─────────────────────────────────────
    best_acc   = 0.0
    best_preds = best_targets = []
    tl_log = []; vl_log = []; ta_log = []; va_log = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ta            = run_train(model, tr_ld, criterion, optimizer)
        vl, va, preds, gt = run_eval(model, vl_ld, criterion)
        scheduler.step()
        elapsed = time.time() - t0
        tl_log.append(tl); vl_log.append(vl)
        ta_log.append(ta); va_log.append(va)

        star = "  *** BEST ***" if va > best_acc else ""
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Train {ta:.1f}%  Val {va:.1f}%  "
              f"Loss {tl:.4f}/{vl:.4f}  ({elapsed:.0f}s){star}")

        if va > best_acc:
            best_acc     = va
            best_preds   = preds
            best_targets = gt
            acc_str  = str(round(best_acc)).zfill(2)
            mname    = f"model_{acc_str}_acc_image_resnext_data.pt"
            mpath    = os.path.join(MODELS_DIR, mname)
            torch.save(model.state_dict(), mpath)
            print(f"  >> Saved: {mname}")

    # ── Step 7: final report ──────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best Validation Accuracy : {best_acc:.2f}%")
    print(f"  Model saved to           : {MODELS_DIR}")
    print(f"{'='*55}\n")

    save_curves(tl_log, vl_log, ta_log, va_log, BASE)

    # Confusion matrix
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(best_targets, best_preds)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (Best Val Acc: {best_acc:.1f}%)')
        plt.tight_layout()
        cpath = os.path.join(BASE, 'image_confusion_matrix.png')
        plt.savefig(cpath, dpi=150); plt.close()
        print(f"[INFO] Confusion matrix -> {cpath}")
        tn,fp,fn,tp = cm.ravel()
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
        print(f"  Precision={p*100:.1f}%  Recall={r*100:.1f}%  "
              f"F1={2*p*r/(p+r+1e-8)*100:.1f}%")
    except ImportError:
        pass

    print("\n[DONE] Upload an image at http://127.0.0.1:8000/ to test the model.")


if __name__ == '__main__':
    main()
