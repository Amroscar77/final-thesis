"""
Deepfake Detection V2 — Training Script
Architecture: EfficientNet-B4 + DCT + BiGRU + Temporal Attention
"""

import os
import sys
import argparse
import json
import random
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import cv2

# ─── Attempt imports ───
try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
    print("[WARN] facenet_pytorch not installed. Using cv2 cascade for face detection.")

try:
    import timm
except ImportError:
    print("[ERROR] timm is required. Install with: pip install timm")
    sys.exit(1)

try:
    from scipy.fftpack import dct as scipy_dct
except ImportError:
    scipy_dct = None


# ═══════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════

class DCTFrequencyBranch(nn.Module):
    def __init__(self, out_features=256):
        super().__init__()
        self.freq_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_features)

    def _apply_dct(self, x):
        if scipy_dct is not None:
            x_np = x.detach().cpu().numpy()
            dct_result = np.zeros_like(x_np)
            for b in range(x_np.shape[0]):
                for c in range(x_np.shape[1]):
                    row_dct = scipy_dct(x_np[b, c], type=2, axis=0, norm='ortho')
                    dct_result[b, c] = scipy_dct(row_dct, type=2, axis=1, norm='ortho')
            return torch.from_numpy(dct_result).to(x.device, dtype=x.dtype)
        else:
            x_fft = torch.fft.fft2(x, norm='ortho')
            return x_fft.real

    def forward(self, x):
        dct_x = self._apply_dct(x)
        dct_x = torch.log1p(torch.abs(dct_x))
        features = self.freq_cnn(dct_x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, gru_output):
        weights = F.softmax(self.attention(gru_output), dim=1)
        context = torch.sum(gru_output * weights, dim=1)
        return context, weights


class DeepfakeDetectorV2(nn.Module):
    def __init__(self, num_classes=2, freq_dim=256, fused_dim=512,
                 gru_hidden=512, gru_layers=2, dropout=0.5):
        super().__init__()

        self.spatial_backbone = timm.create_model(
            'efficientnet_b4', pretrained=True,
            features_only=False, num_classes=0, global_pool='avg',
        )
        spatial_dim = self.spatial_backbone.num_features

        self.freq_branch = DCTFrequencyBranch(out_features=freq_dim)
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, fused_dim),
            nn.BatchNorm1d(fused_dim), nn.ReLU(True),
            nn.Dropout(dropout * 0.5),
        )
        self.gru = nn.GRU(fused_dim, gru_hidden, gru_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0)
        self.attention = TemporalAttention(gru_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(gru_hidden * 2, 256),
            nn.ReLU(True), nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )
        self.image_classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fused_dim, 256),
            nn.ReLU(True), nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def _extract_frame_features(self, x):
        spatial = self.spatial_backbone(x)
        freq = self.freq_branch(x)
        combined = torch.cat([spatial, freq], dim=1)
        return self.fusion(combined)

    def forward(self, x, mode='video'):
        if mode == 'image' or x.dim() == 4:
            fused = self._extract_frame_features(x)
            return self.image_classifier(fused), {}

        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        spatial_feats = self.spatial_backbone(x_flat)
        freq_feats = self.freq_branch(x_flat)
        combined = torch.cat([spatial_feats, freq_feats], dim=1)
        fused = self.fusion(combined).view(B, T, -1)
        gru_out, _ = self.gru(fused)
        context, attn_weights = self.attention(gru_out)
        logits = self.classifier(context)
        return logits, {'attention_weights': attn_weights.detach()}


# ═══════════════════════════════════════════════
# Face Detection & Extraction
# ═══════════════════════════════════════════════

class FaceExtractor:
    """Extract and align faces from video frames using MTCNN."""

    def __init__(self, device='cuda', face_size=300, margin=40):
        self.face_size = face_size
        self.margin = margin
        self.device = device

        if HAS_MTCNN:
            self.detector = MTCNN(
                image_size=face_size,
                margin=margin,
                keep_all=False,
                device=device,
                post_process=False,  # Return raw pixel values
            )
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)

    def extract_face(self, frame_rgb):
        """Extract a single face from an RGB frame."""
        if HAS_MTCNN:
            try:
                face = self.detector(frame_rgb)
                if face is not None:
                    # MTCNN returns (C, H, W) tensor in [0, 255]
                    return face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            except Exception:
                pass
        else:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                m = self.margin
                y1 = max(0, y - m)
                y2 = min(frame_rgb.shape[0], y + h + m)
                x1 = max(0, x - m)
                x2 = min(frame_rgb.shape[1], x + w + m)
                face = frame_rgb[y1:y2, x1:x2]
                face = cv2.resize(face, (self.face_size, self.face_size))
                return face
        return None


# ═══════════════════════════════════════════════
# DFDC Dataset
# ═══════════════════════════════════════════════

class DFDCDataset(Dataset):
    """
    DFDC dataset loader.
    
    Expected directory structure:
        data_dir/
            dfdc_train_part_0/
                metadata.json
                aaavqmfxwx.mp4
                ...
            dfdc_train_part_1/
                ...
            ...
    """

    def __init__(self, data_dir, sequence_length=20, transform=None,
                 max_videos=None, face_extractor=None, split='train'):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.face_extractor = face_extractor
        self.split = split

        # Load all metadata
        self.videos = []
        self.labels = {}

        parts = sorted(self.data_dir.glob('dfdc_train_part_*'))
        if not parts:
            # Try flat structure
            parts = [self.data_dir]

        for part_dir in parts:
            meta_path = part_dir / 'metadata.json'
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                for video_name, info in metadata.items():
                    video_path = part_dir / video_name
                    if video_path.exists():
                        label = 0 if info.get('label', 'FAKE') == 'FAKE' else 1
                        self.videos.append(str(video_path))
                        self.labels[str(video_path)] = label

        # Shuffle and limit
        random.shuffle(self.videos)
        if max_videos:
            self.videos = self.videos[:max_videos]

        # Train/val split
        split_idx = int(len(self.videos) * 0.85)
        if split == 'train':
            self.videos = self.videos[:split_idx]
        else:
            self.videos = self.videos[split_idx:]

        print(f"[{split.upper()}] Loaded {len(self.videos)} videos")

        # Count labels
        label_counts = defaultdict(int)
        for v in self.videos:
            label_counts[self.labels[v]] += 1
        print(f"  FAKE: {label_counts[0]}, REAL: {label_counts[1]}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[video_path]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames evenly across the video
        if total_frames <= self.sequence_length:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1,
                                  self.sequence_length, dtype=int).tolist()

        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract face
            if self.face_extractor:
                face = self.face_extractor.extract_face(rgb_frame)
                if face is not None:
                    rgb_frame = face

            if self.transform:
                rgb_frame = self.transform(rgb_frame)
                frames.append(rgb_frame)

            if len(frames) >= self.sequence_length:
                break

        cap.release()

        # Pad if necessary
        if len(frames) == 0:
            # Return a dummy tensor
            frames = [torch.zeros(3, 300, 300)] * self.sequence_length

        while len(frames) < self.sequence_length:
            frames.append(frames[-1].clone())

        frames = torch.stack(frames[:self.sequence_length])  # (T, C, H, W)
        return frames, label


# ═══════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (frames, labels) in enumerate(dataloader):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits, _ = model(frames, mode='video')
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for frames, labels in dataloader:
        frames = frames.to(device)
        labels = labels.to(device)

        with autocast():
            logits, _ = model(frames, mode='video')
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detector V2')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to DFDC dataset directory')
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq_length', type=int, default=20)
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Max videos to use (for quick testing)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze EfficientNet backbone (faster, less VRAM)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Transforms ──
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Face Extractor ──
    face_extractor = FaceExtractor(device=str(device), face_size=300)

    # ── Datasets ──
    print("\n[INFO] Loading training data...")
    train_dataset = DFDCDataset(
        args.data_dir, sequence_length=args.seq_length,
        transform=train_transform, max_videos=args.max_videos,
        face_extractor=face_extractor, split='train',
    )

    print("\n[INFO] Loading validation data...")
    val_dataset = DFDCDataset(
        args.data_dir, sequence_length=args.seq_length,
        transform=val_transform, max_videos=args.max_videos,
        face_extractor=face_extractor, split='val',
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── Model ──
    print("\n[INFO] Initializing model...")
    model = DeepfakeDetectorV2(num_classes=2)

    if args.freeze_backbone:
        print("  [INFO] Freezing EfficientNet backbone")
        for param in model.spatial_backbone.parameters():
            param.requires_grad = False

    if args.resume:
        print(f"  [INFO] Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.5]).to(device)
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    scaler = GradScaler()

    best_val_acc = 0
    patience = 7
    patience_counter = 0

    print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.epochs):
        start_time = time.time()

        print(f"\nEpoch {epoch+1}/{args.epochs} (lr: {optimizer.param_groups[0]['lr']:.6f})")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            acc_str = int(val_acc * 100)
            save_name = f"model_{acc_str}_acc_{args.seq_length}_frames_v2_effnet_gru.pt"
            save_path = os.path.join(args.output_dir, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"  [SAVED] {save_name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, ckpt_path)
            print(f"  [CKPT] checkpoint_epoch_{epoch+1}.pt")

    print("\n" + "=" * 60)
    print(f"[INFO] Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"[INFO] Models saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
