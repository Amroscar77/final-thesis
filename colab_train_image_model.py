# ============================================================
#  DeepScan — Image Deepfake Detector — Google Colab Training
#  Copy each section into a separate Colab cell and run top-to-bottom.
#  Requires: Runtime → Change runtime type → T4 GPU
# ============================================================


# ── CELL 1: Install dependencies ────────────────────────────
# !pip install kagglehub -q


# ── CELL 2: Kaggle authentication ───────────────────────────
# Option A — upload your kaggle.json token file:
#   from google.colab import files
#   files.upload()          # choose kaggle.json
#   import os, shutil
#   shutil.move("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
#   os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
#
# Option B — paste token directly (replace with your values):
#   import os, json
#   os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
#   token = {"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}
#   with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
#       json.dump(token, f)
#   os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)


# ── CELL 3: Download dataset ─────────────────────────────────
"""
import kagglehub
DATA_ROOT = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
print("Dataset path:", DATA_ROOT)
import os
TRAIN_REAL = os.path.join(DATA_ROOT, "Dataset", "Train", "Real")
TRAIN_FAKE = os.path.join(DATA_ROOT, "Dataset", "Train", "Fake")
VAL_REAL   = os.path.join(DATA_ROOT, "Dataset", "Validation", "Real")
VAL_FAKE   = os.path.join(DATA_ROOT, "Dataset", "Validation", "Fake")
print("Train Real:", len(os.listdir(TRAIN_REAL)))
print("Train Fake:", len(os.listdir(TRAIN_FAKE)))
print("Val   Real:", len(os.listdir(VAL_REAL)))
print("Val   Fake:", len(os.listdir(VAL_FAKE)))
"""


# ── CELL 4: Full training script ─────────────────────────────
COLAB_TRAINING_SCRIPT = '''
import os, random, time, gc
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ── Config ───────────────────────────────────────────────────
TRAIN_REAL = os.path.join(DATA_ROOT, "Dataset", "Train", "Real")
TRAIN_FAKE = os.path.join(DATA_ROOT, "Dataset", "Train", "Fake")
VAL_REAL   = os.path.join(DATA_ROOT, "Dataset", "Validation", "Real")
VAL_FAKE   = os.path.join(DATA_ROOT, "Dataset", "Validation", "Fake")

SAMPLES_PER_CLASS = 20000   # 20K real + 20K fake = 40K training images
                             # Set to None to use ALL 70K per class (slower)
VAL_SAMPLES       = 5000    # validation samples per class
IMG_SIZE          = 224
BATCH_SIZE        = 64      # GPU can handle 64
EPOCHS            = 15
LR                = 1e-4
WEIGHT_DECAY      = 1e-4
SAVE_DIR          = "/content"   # change to "/content/drive/MyDrive" to save to Google Drive
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Transforms ───────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Dataset ───────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder, max_n=None):
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    if max_n and len(files) > max_n:
        files = random.sample(files, max_n)
    return files

class FaceDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]
        except Exception:
            blank = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            return self.transform(blank), self.labels[idx]

# Build train set
real_train = list_images(TRAIN_REAL, SAMPLES_PER_CLASS)
fake_train = list_images(TRAIN_FAKE, SAMPLES_PER_CLASS)
train_paths  = real_train + fake_train
train_labels = [1] * len(real_train) + [0] * len(fake_train)
combined = list(zip(train_paths, train_labels))
random.shuffle(combined)
train_paths, train_labels = zip(*combined)
print(f"Train: {len(real_train)} real + {len(fake_train)} fake = {len(train_paths)} total")

# Build val set
real_val  = list_images(VAL_REAL, VAL_SAMPLES)
fake_val  = list_images(VAL_FAKE, VAL_SAMPLES)
val_paths  = real_val + fake_val
val_labels = [1] * len(real_val) + [0] * len(fake_val)
print(f"Val  : {len(real_val)} real + {len(fake_val)} fake = {len(val_paths)} total")

train_ds = FaceDataset(list(train_paths), list(train_labels), train_tf)
val_ds   = FaceDataset(val_paths, val_labels, val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────
# Architecture MUST match build_image_model() in views.py:
#   keys: conv1.*, bn1.*, layer1-4.*, fc.1.* (Linear 2048->512), fc.4.* (Linear 512->2)
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.25),
    nn.Linear(512, 2),
)
model = model.to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Training ──────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
best_path    = ""

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ──
    model.train()
    correct = total = 0
    run_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
    train_acc  = correct / total * 100
    train_loss = run_loss / total
    scheduler.step()

    # ── Validate ──
    model.eval()
    correct = total = 0
    run_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            run_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
    val_acc  = correct / total * 100
    val_loss = run_loss / total
    elapsed  = int(time.time() - t0)

    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        acc_int      = int(round(val_acc))
        save_name    = f"model_{acc_int}_acc_image_resnext_data.pt"
        best_path    = os.path.join(SAVE_DIR, save_name)
        torch.save(model.state_dict(), best_path)
        marker = f"  *** BEST -> {save_name}"

    print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
          f"Train {train_acc:.1f}%  Val {val_acc:.1f}%  "
          f"Loss {train_loss:.4f}/{val_loss:.4f}  ({elapsed}s){marker}")

print(f"\\nDone. Best Val Accuracy: {best_val_acc:.2f}%")
print(f"Saved: {best_path}")
'''

# Print the script so you can copy it into a Colab cell
print(COLAB_TRAINING_SCRIPT)


# ── CELL 5: Download the model to your PC ────────────────────
"""
from google.colab import files
files.download(best_path)   # downloads the .pt file to your browser
"""


# ── CELL 6 (optional): Save to Google Drive instead ──────────
"""
from google.colab import drive
drive.mount("/content/drive")
# Then set SAVE_DIR = "/content/drive/MyDrive" before training
"""
