"""
Image-Only Deepfake Detection — Training Script
Trains a CNN (ResNeXt-50 or EfficientNet-B4) on still images.

Usage:
  python train_image_model.py --data_dir ./data
  python train_image_model.py --csv ./labels.csv
  python train_image_model.py --data_dir ./data --model efficientnet --epochs 30
"""

import os
import sys
import glob
import time
import random
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# ── Optional heavy imports ──────────────────────────────────────
try:
    from facenet_pytorch import MTCNN as _MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn/seaborn not found. Confusion matrix will be skipped.")

# ── Supported image extensions ──────────────────────────────────
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

# ═══════════════════════════════════════════════════════════════
# Face Detector (singleton per process)
# ═══════════════════════════════════════════════════════════════
_mtcnn_instance = None
_cascade_instance = None

def get_face_crop(img_rgb, face_size=224, padding=20):
    """
    Try MTCNN first, fall back to Haar cascade, then return full image.
    img_rgb : HxWx3 numpy array in RGB order.
    Returns  : cropped face as HxWx3 numpy array.
    """
    global _mtcnn_instance, _cascade_instance

    # --- MTCNN ---
    if HAS_MTCNN:
        if _mtcnn_instance is None:
            _mtcnn_instance = _MTCNN(
                image_size=face_size, margin=40,
                keep_all=False, device='cpu', post_process=False
            )
        try:
            boxes, _ = _mtcnn_instance.detect(Image.fromarray(img_rgb))
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                h, w = img_rgb.shape[:2]
                x1 = max(0, x1 - padding); y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding); y2 = min(h, y2 + padding)
                face = img_rgb[y1:y2, x1:x2]
                if face.size > 0:
                    return face
        except Exception:
            pass

    # --- Haar cascade fallback ---
    if _cascade_instance is None:
        _cascade_instance = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _cascade_instance.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        img_h, img_w = img_rgb.shape[:2]
        x1 = max(0, x - padding);     y1 = max(0, y - padding)
        x2 = min(img_w, x+w+padding); y2 = min(img_h, y+h+padding)
        face = img_rgb[y1:y2, x1:x2]
        if face.size > 0:
            return face

    return img_rgb  # fallback: full image


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════
class ImageDeepfakeDataset(Dataset):
    """
    Loads images from (path, label) lists.
    label: 0 = FAKE, 1 = REAL
    Optionally crops the face region before applying transforms.
    """

    def __init__(self, paths, labels, transform=None,
                 crop_face=True, face_size=224):
        assert len(paths) == len(labels), "paths and labels must be same length"
        self.paths     = paths
        self.labels    = labels
        self.transform = transform
        self.crop_face = crop_face
        self.face_size = face_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]

        img = cv2.imread(path)
        if img is None:
            # Return blank tensor on unreadable file
            dummy = np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8)
            return self.transform(dummy) if self.transform else \
                   torch.zeros(3, self.face_size, self.face_size), label

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.crop_face:
            img = get_face_crop(img, face_size=self.face_size)

        if self.transform:
            img = self.transform(img)

        return img, label


# ═══════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════
class ResNeXtImageModel(nn.Module):
    """
    ResNeXt-50 (32x4d) backbone with a custom 2-class head.
    No LSTM — pure spatial classification for still images.
    Compatible with the existing V1 model weight format.
    """
    def __init__(self, num_classes=2, dropout=0.5, freeze_backbone=False):
        super().__init__()
        backbone = models.resnext50_32x4d(pretrained=True)
        # Remove the original FC layer; keep everything up to AdaptiveAvgPool
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
        x = self.features(x)       # (B, 2048, 1, 1)
        return self.classifier(x)  # (B, num_classes)


class EfficientNetImageModel(nn.Module):
    """EfficientNet-B4 via timm with a custom 2-class head."""
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError(
                "timm is not installed. Run: pip install timm>=0.9.0"
            )
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=True,
            num_classes=0, global_pool='avg'
        )
        in_features = self.backbone.num_features  # 1792

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)       # (B, 1792)
        return self.classifier(x)  # (B, num_classes)


# ═══════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════
def load_from_folder(root_dir):
    """
    Expects:
        root_dir/real/  -> label 1
        root_dir/fake/  -> label 0
    Searches recursively for all supported image extensions.
    """
    paths, labels = [], []
    for label_name, label_val in [('real', 1), ('fake', 0)]:
        folder = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue
        for ext in IMG_EXTS:
            for p in glob.glob(os.path.join(folder, '**', f'*{ext}'),
                               recursive=True):
                paths.append(p)
                labels.append(label_val)
            for p in glob.glob(os.path.join(folder, '**', f'*{ext.upper()}'),
                               recursive=True):
                paths.append(p)
                labels.append(label_val)
    return paths, labels


def load_from_csv(csv_path):
    """
    CSV must have columns: path, label
    label: REAL/FAKE (strings) or 1/0 (integers)
    """
    df = pd.read_csv(csv_path)
    if 'path' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have 'path' and 'label' columns.")
    paths  = df['path'].tolist()
    labels = []
    for l in df['label'].tolist():
        if isinstance(l, str):
            labels.append(1 if l.strip().upper() == 'REAL' else 0)
        else:
            labels.append(int(l))
    return paths, labels


# ═══════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def calculate_accuracy(outputs, targets):
    _, pred = outputs.topk(1, 1, True)
    correct = pred.t().eq(targets.view(1, -1)).float().sum().item()
    return 100.0 * correct / targets.size(0)


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    losses = AverageMeter()
    accs   = AverageMeter()
    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device).long()
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(calculate_accuracy(outputs, targets), inputs.size(0))
    return losses.avg, accs.avg


def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses      = AverageMeter()
    accs        = AverageMeter()
    all_preds   = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device).long()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accs.update(calculate_accuracy(outputs, targets), inputs.size(0))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
    return losses.avg, accs.avg, all_preds, all_targets


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════
def save_training_curves(train_losses, val_losses, train_accs, val_accs,
                          save_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'g-o', label='Train Loss')
    ax1.plot(epochs, val_losses,   'b-o', label='Val Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'g-o', label='Train Accuracy')
    ax2.plot(epochs, val_accs,   'b-o', label='Val Accuracy')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved -> {out}")


def save_confusion_matrix(all_targets, all_preds, save_dir):
    if not HAS_SKLEARN:
        return
    cm  = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'],
                annot_kws={"size": 16})
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('Actual label',    fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    out = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(out, dpi=150)
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    print(f"\n{'='*45}")
    print(f"  True  Positive (Real -> Real) : {tp}")
    print(f"  True  Negative (Fake -> Fake) : {tn}")
    print(f"  False Positive (Fake -> Real) : {fp}")
    print(f"  False Negative (Real -> Fake) : {fn}")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"{'='*45}")
    print(f"[INFO] Confusion matrix saved -> {out}\n")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Train an image-only deepfake detector'
    )
    # Data
    parser.add_argument('--data_dir', type=str, default=None,
        help='Root folder with real/ and fake/ subfolders')
    parser.add_argument('--csv', type=str, default=None,
        help='CSV file with columns: path, label')
    # Model
    parser.add_argument('--model', type=str, default='resnext',
        choices=['resnext', 'efficientnet'],
        help='Backbone: resnext (default) or efficientnet')
    parser.add_argument('--freeze', action='store_true',
        help='Freeze backbone and only train the classifier head')
    # Training
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--img_size',   type=int,   default=224)
    parser.add_argument('--val_split',  type=float, default=0.2,
        help='Fraction of data to use for validation (default: 0.2)')
    parser.add_argument('--no_face_crop', action='store_true',
        help='Skip face detection and use full images')
    parser.add_argument('--workers', type=int, default=0,
        help='DataLoader workers (0 = single-process, safe on Windows)')
    parser.add_argument('--seed',    type=int, default=42)
    # Output
    parser.add_argument('--output_dir', type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            'Deepfake_detection_using_deep_learning',
            'Django Application', 'models'
        ),
        help='Directory to save the trained model and plots')
    args = parser.parse_args()

    # ── Reproducibility ─────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[INFO] Device      : {device}")
    print(f"[INFO] PyTorch     : {torch.__version__}")

    # ── Load data ───────────────────────────────────────────────
    if args.data_dir:
        paths, labels = load_from_folder(args.data_dir)
    elif args.csv:
        paths, labels = load_from_csv(args.csv)
    else:
        print("\n[ERROR] Provide --data_dir or --csv\n")
        parser.print_help()
        sys.exit(1)

    if len(paths) == 0:
        print("[ERROR] No images found. Check your --data_dir or --csv.")
        sys.exit(1)

    n_real = sum(l == 1 for l in labels)
    n_fake = sum(l == 0 for l in labels)
    print(f"[INFO] Total images: {len(paths)}  (Real: {n_real}, Fake: {n_fake})")

    # ── Shuffle & split ──────────────────────────────────────────
    combined = list(zip(paths, labels))
    random.shuffle(combined)
    paths, labels = [x[0] for x in combined], [x[1] for x in combined]

    split        = int(len(paths) * (1 - args.val_split))
    train_paths  = paths[:split];  train_labels = labels[:split]
    val_paths    = paths[split:];  val_labels   = labels[split:]
    print(f"[INFO] Train       : {len(train_paths)} images")
    print(f"[INFO] Validation  : {len(val_paths)} images")

    # ── Transforms ──────────────────────────────────────────────
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.1),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        norm,
    ])

    crop = not args.no_face_crop
    train_ds = ImageDeepfakeDataset(train_paths, train_labels,
                                     transform=train_tf,
                                     crop_face=crop,
                                     face_size=args.img_size)
    val_ds   = ImageDeepfakeDataset(val_paths,   val_labels,
                                     transform=val_tf,
                                     crop_face=crop,
                                     face_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=(device == 'cuda'))

    # ── Model ────────────────────────────────────────────────────
    print(f"[INFO] Architecture: {args.model}")
    if args.model == 'resnext':
        model = ResNeXtImageModel(num_classes=2, freeze_backbone=args.freeze)
    else:
        model = EfficientNetImageModel(num_classes=2)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params:,}")
    if args.freeze:
        print("[INFO] Backbone frozen — training classifier head only")

    # ── Optimizer / scheduler / loss ─────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    # ── Training loop ─────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    best_acc     = 0.0
    best_preds   = []
    best_targets = []
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print(f"\n{'='*55}")
    print(f"  Starting training")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Image size  : {args.img_size}x{args.img_size}")
    print(f"  Face crop   : {crop}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"{'='*55}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tl, ta = train_epoch(model, train_loader, criterion,
                              optimizer, device, scaler)
        vl, va, preds, targets = eval_epoch(model, val_loader,
                                             criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta);   val_accs.append(va)

        marker = "  *** BEST ***" if va > best_acc else ""
        print(f"Epoch [{epoch:02d}/{args.epochs}]  "
              f"Train Loss: {tl:.4f}  Train Acc: {ta:.1f}%  |  "
              f"Val Loss: {vl:.4f}  Val Acc: {va:.1f}%  "
              f"({elapsed:.0f}s){marker}")

        if va > best_acc:
            best_acc     = va
            best_preds   = preds
            best_targets = targets
            acc_str      = str(round(best_acc)).zfill(2)
            model_name   = f"model_{acc_str}_acc_image_{args.model}_data.pt"
            save_path    = os.path.join(args.output_dir, model_name)
            torch.save(model.state_dict(), save_path)
            print(f"  >> Saved: {model_name}")

    # ── Final report ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training complete")
    print(f"  Best Validation Accuracy : {best_acc:.2f}%")
    print(f"  Saved model name         : model_{str(round(best_acc)).zfill(2)}"
          f"_acc_image_{args.model}_data.pt")
    print(f"{'='*55}\n")

    save_training_curves(train_losses, val_losses,
                          train_accs,   val_accs,
                          args.output_dir)
    save_confusion_matrix(best_targets, best_preds, args.output_dir)

    print("[INFO] Done. The model is ready for use in the DeepScan web app.")
    print("[INFO] Upload an image at http://127.0.0.1:8000/ to test it.\n")


if __name__ == '__main__':
    main()
