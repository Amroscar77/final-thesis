"""
Deepfake Detection Model V2 — Modern Architecture (2025/2026)

Architecture:
    - EfficientNet-B4 spatial backbone (pretrained on ImageNet)
    - DCT frequency-domain analysis branch
    - Feature fusion (spatial + frequency)
    - Bidirectional GRU with temporal attention
    - Binary classification head (REAL / FAKE)

Improvements over V1 (ResNeXt-50 + LSTM):
    - 300x300 input (vs 112x112) — catches finer artifacts
    - Frequency analysis detects GAN/diffusion spectral fingerprints
    - BiGRU captures both forward & backward temporal patterns
    - Attention mechanism highlights most suspicious frames
    - Grad-CAM compatible for explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import timm
except ImportError:
    timm = None

try:
    from scipy.fftpack import dct as scipy_dct
except ImportError:
    scipy_dct = None


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
IM_SIZE_V2 = 300
MEAN_V2 = [0.485, 0.456, 0.406]
STD_V2 = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# Frequency Analysis Branch (DCT)
# ─────────────────────────────────────────────
class DCTFrequencyBranch(nn.Module):
    """
    Extracts frequency-domain features using DCT (Discrete Cosine Transform).
    GAN-generated faces leave distinctive spectral fingerprints that are
    invisible in pixel space but obvious in the frequency domain.
    """

    def __init__(self, out_features=256):
        super().__init__()
        # Process DCT spectrum with lightweight CNN
        self.freq_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_features)

    def _apply_dct(self, x):
        """Apply DCT to input tensor (batch of images)."""
        # x shape: (B, C, H, W)
        if scipy_dct is not None:
            x_np = x.detach().cpu().numpy()
            # Apply 2D DCT to each channel
            dct_result = np.zeros_like(x_np)
            for b in range(x_np.shape[0]):
                for c in range(x_np.shape[1]):
                    # Row-wise DCT then column-wise DCT
                    row_dct = scipy_dct(x_np[b, c], type=2, axis=0, norm='ortho')
                    dct_result[b, c] = scipy_dct(row_dct, type=2, axis=1, norm='ortho')
            return torch.from_numpy(dct_result).to(x.device, dtype=x.dtype)
        else:
            # Fallback: use FFT-based approximation
            x_fft = torch.fft.fft2(x, norm='ortho')
            return x_fft.real

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) face image tensor
        Returns:
            freq_features: (B, out_features)
        """
        dct_x = self._apply_dct(x)
        # Log-scale for better dynamic range
        dct_x = torch.log1p(torch.abs(dct_x))
        features = self.freq_cnn(dct_x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


# ─────────────────────────────────────────────
# Temporal Attention Module
# ─────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Learns which frames in a video sequence are most important for
    the real/fake decision. Suspicious frames get higher weights.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, gru_output):
        """
        Args:
            gru_output: (B, T, hidden_dim) — GRU outputs for all timesteps
        Returns:
            context: (B, hidden_dim) — attention-weighted representation
            weights: (B, T, 1) — attention weights per frame
        """
        weights = self.attention(gru_output)  # (B, T, 1)
        weights = F.softmax(weights, dim=1)  # normalize across time
        context = torch.sum(gru_output * weights, dim=1)  # (B, hidden_dim)
        return context, weights


# ─────────────────────────────────────────────
# Main Model: DeepfakeDetectorV2
# ─────────────────────────────────────────────
class DeepfakeDetectorV2(nn.Module):
    """
    Modern deepfake detection model combining:
    1. EfficientNet-B4 for spatial feature extraction
    2. DCT frequency analysis for spectral artifact detection
    3. Bidirectional GRU for temporal modeling
    4. Temporal attention for frame importance weighting
    """

    def __init__(
        self,
        num_classes=2,
        spatial_dim=1792,      # EfficientNet-B4 feature dim
        freq_dim=256,          # Frequency branch output dim
        fused_dim=512,         # Dimension after fusion
        gru_hidden=512,        # GRU hidden size
        gru_layers=2,          # Number of GRU layers
        dropout=0.5,
    ):
        super().__init__()

        # ── Spatial Backbone: EfficientNet-B4 ──
        if timm is not None:
            self.spatial_backbone = timm.create_model(
                'efficientnet_b4',
                pretrained=True,
                features_only=False,
                num_classes=0,  # Remove classifier, keep features
                global_pool='avg',
            )
            # Get actual feature dim from the model
            spatial_dim = self.spatial_backbone.num_features
        else:
            # Fallback to torchvision
            from torchvision import models
            eff = models.efficientnet_b4(weights='IMAGENET1K_V1')
            self.spatial_backbone = nn.Sequential(*list(eff.children())[:-1])
            spatial_dim = 1792
            self.spatial_backbone.add_module('flatten', nn.Flatten())

        # ── Frequency Branch ──
        self.freq_branch = DCTFrequencyBranch(out_features=freq_dim)

        # ── Feature Fusion ──
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        # ── Temporal Model: Bidirectional GRU ──
        self.gru = nn.GRU(
            input_size=fused_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        # ── Temporal Attention ──
        self.attention = TemporalAttention(gru_hidden * 2)  # *2 for bidirectional

        # ── Classification Head ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # ── For single image mode (no temporal) ──
        self.image_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # Store the last feature map for Grad-CAM
        self.last_fmap = None
        self.last_attention_weights = None

    def _extract_frame_features(self, x):
        """
        Extract spatial + frequency features for a single frame.
        Args:
            x: (B, C, H, W) single frame
        Returns:
            fused: (B, fused_dim) fused features
        """
        # Spatial features
        spatial = self.spatial_backbone(x)  # (B, spatial_dim)

        # Frequency features
        freq = self.freq_branch(x)  # (B, freq_dim)

        # Fuse
        combined = torch.cat([spatial, freq], dim=1)  # (B, spatial_dim + freq_dim)
        fused = self.fusion(combined)  # (B, fused_dim)

        return fused

    def forward(self, x, mode='video'):
        """
        Args:
            x: For video: (B, T, C, H, W) — batch of frame sequences
               For image: (B, C, H, W) — batch of single images
            mode: 'video' or 'image'
        Returns:
            logits: (B, num_classes)
            extras: dict with attention_weights, feature_maps for explainability
        """
        extras = {}

        if mode == 'image' or (x.dim() == 4):
            # ─── Image Mode ───
            fused = self._extract_frame_features(x)
            logits = self.image_classifier(fused)
            extras['mode'] = 'image'
            return logits, extras

        # ─── Video Mode ───
        B, T, C, H, W = x.shape

        # Extract features for all frames
        x_flat = x.view(B * T, C, H, W)  # (B*T, C, H, W)

        # Spatial features (without BatchNorm issues)
        spatial_feats = self.spatial_backbone(x_flat)  # (B*T, spatial_dim)
        freq_feats = self.freq_branch(x_flat)  # (B*T, freq_dim)

        combined = torch.cat([spatial_feats, freq_feats], dim=1)

        # Handle BatchNorm in fusion layer
        fused = self.fusion(combined)  # (B*T, fused_dim)
        fused = fused.view(B, T, -1)  # (B, T, fused_dim)

        # Temporal modeling
        gru_out, _ = self.gru(fused)  # (B, T, gru_hidden*2)

        # Attention-weighted aggregation
        context, attn_weights = self.attention(gru_out)  # (B, gru_hidden*2)
        self.last_attention_weights = attn_weights

        extras['attention_weights'] = attn_weights.detach()
        extras['mode'] = 'video'

        # Classification
        logits = self.classifier(context)  # (B, num_classes)

        return logits, extras

    def get_attention_weights(self):
        """Return the attention weights from the last forward pass."""
        return self.last_attention_weights


# ─────────────────────────────────────────────
# Legacy Model (V1) — kept for backward compatibility
# ─────────────────────────────────────────────
class LegacyModel(nn.Module):
    """Original ResNeXt-50 + LSTM model (V1) for backward compatibility."""

    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1,
                 hidden_dim=2048, bidirectional=False):
        super().__init__()
        from torchvision import models
        model = models.resnext50_32x4d(weights=None)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────
def get_v2_transforms(im_size=IM_SIZE_V2, is_training=False):
    """Get transforms for V2 model."""
    from torchvision import transforms

    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_V2, STD_V2),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_V2, STD_V2),
        ])


def load_model_v2(model_path, device='cpu', num_classes=2):
    """Load a trained V2 model from checkpoint."""
    model = DeepfakeDetectorV2(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    # Handle DataParallel saved models
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_legacy_model(model_path, device='cpu', num_classes=2):
    """Load a legacy V1 model."""
    model = LegacyModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
