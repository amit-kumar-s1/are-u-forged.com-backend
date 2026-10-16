"""
models/deepfake_model.py
─────────────────────────
Architecture:
  EfficientNetV2-S (backbone)
       ↓
  Linear projection → d_model
       ↓
  Positional Encoding
       ↓
  Transformer Encoder (4 layers, 8 heads)
       ↓
  Attention Pooling  (collapses time dimension)
       ↓
  Dropout → Linear → Logit
"""

import math
import logging
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# type alias hack for clarity

Tuple_nn_int = Tuple[nn.Module, int]

from app.detectors.video.config import model_cfg, ModelConfig

cfg = type("TrainingConfig", (), {"model": model_cfg})()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                         # (L, D)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                                       # (1, L, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Attention Pooling
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Learns a query vector that attends over the temporal dimension.
    Input:  (B, T, D)
    Output: (B, D)
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    (B, T, D)
        mask: (B, T) bool — True = padding (ignored)
        """
        scores = self.net(x).squeeze(-1)                # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled  = (weights * x).sum(dim=1)                 # (B, D)
        return pooled


# ─────────────────────────────────────────────────────────────────────────────
# 3. EfficientNetV2-S Backbone
# ─────────────────────────────────────────────────────────────────────────────

def _build_backbone(pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Returns (feature_extractor, out_channels)."""
    weights = tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = tv_models.efficientnet_v2_s(weights=weights)
    # Remove the classifier head, keep features only
    out_channels = backbone.classifier[1].in_features   # 1280
    backbone.classifier = nn.Identity()
    backbone.avgpool    = nn.Identity()
    # We'll do our own pooling; keep features spatial then flatten
    return backbone, out_channels





# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Model
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    Binary deepfake detector.
    Input:  (B, T, C, H, W)  — batch of video clips
    Output: (B,)              — raw logits (apply sigmoid for probability)
    """

    def __init__(self, model_cfg: ModelConfig = cfg.model):
        super().__init__()
        self.model_cfg = model_cfg

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone, backbone_out = _build_backbone(model_cfg.backbone_pretrained)
        # Spatial pooling: (B*T, C, H', W') → (B*T, C)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # ── Projection ────────────────────────────────────────────────────────
        d = model_cfg.transformer_d_model
        self.proj = nn.Sequential(
            nn.Linear(backbone_out, d),
            nn.LayerNorm(d),
        )

        # ── Positional Encoding ───────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(
            d_model = d,
            max_len = model_cfg.frames_per_clip + 16,
            dropout = model_cfg.transformer_dropout,
        )

        # ── Transformer Encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d,
            nhead           = model_cfg.transformer_nhead,
            dim_feedforward = model_cfg.transformer_dim_ff,
            dropout         = model_cfg.transformer_dropout,
            activation      = "gelu",
            batch_first     = True,   # (B, T, D) convention
            norm_first      = True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer = enc_layer,
            num_layers    = model_cfg.transformer_layers,
            norm          = nn.LayerNorm(d),
        )

        # ── Attention Pooling ─────────────────────────────────────────────────
        self.attn_pool = AttentionPooling(d, model_cfg.attn_pool_hidden)

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(model_cfg.classifier_dropout),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.LayerNorm(d // 2),
            nn.Dropout(model_cfg.classifier_dropout / 2),
            nn.Linear(d // 2, model_cfg.num_classes),
        )

        self._init_weights()
        logger.info(
            f"DeepfakeDetector built: "
            f"{sum(p.numel() for p in self.parameters()) / 1e6:.1f}M params"
        )

    def _init_weights(self):
        for m in [self.proj, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, C, H, W)
        returns: (B, T, backbone_out)
        """
        B, T, C, H, W = frames.shape
        # Merge batch + time
        x = frames.view(B * T, C, H, W)
        x = self.backbone.features(x)         # (B*T, C', H', W')
        x = self.spatial_pool(x)              # (B*T, C', 1, 1)
        x = x.view(B * T, -1)                 # (B*T, C')
        x = x.view(B, T, -1)                  # (B, T, C')
        return x

    def forward(
        self,
        frames: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        frames: (B, T, C, H, W)
        returns: (B,) logits
        """
        # 1. Per-frame CNN features
        feat = self._extract_features(frames)     # (B, T, backbone_out)
        if torch.isnan(feat).any():
            raise RuntimeError("NaN after backbone")

        if torch.isinf(feat).any():
            raise RuntimeError("Inf after backbone")
        
        
        # 2. Project to transformer dim
        feat = self.proj(feat)                    # (B, T, d_model)
        if torch.isnan(feat).any():
            raise RuntimeError("NaN after projection")

        if torch.isinf(feat).any():
            raise RuntimeError("Inf after projection")
        

        # 3. Positional encoding
        feat = self.pos_enc(feat)                 # (B, T, d_model)
        if torch.isnan(feat).any():
            raise RuntimeError("NaN after positional encoding")
        
        if torch.isinf(feat).any():
            raise RuntimeError("Inf after positional encoding")
        

        # 4. Transformer
        feat = self.transformer(feat, src_key_padding_mask=src_key_padding_mask)
        # (B, T, d_model)

        if torch.isnan(feat).any():
            raise RuntimeError("NaN after transformer")

        if torch.isinf(feat).any():
            raise RuntimeError("Inf after transformer")


        # 5. Attention pooling
        pooled = self.attn_pool(feat, mask=src_key_padding_mask)  # (B, d_model)
        if torch.isnan(pooled).any():
            raise RuntimeError("NaN after attention pooling")

        if torch.isinf(pooled).any():
            raise RuntimeError("Inf after attention pooling")
        
        # 6. Classify
        logit = self.classifier(pooled).squeeze(-1)   # (B,)
        if torch.isnan(logit).any():
            raise RuntimeError("NaN after classifier")
    
        if torch.isinf(logit).any():
            raise RuntimeError("Inf after classifier")
        
        return logit

    # ── Parameter groups for differential LR ─────────────────────────────────
    def get_param_groups(self, base_lr: float, backbone_lr_scale: float):
        backbone_ids = set(id(p) for p in self.backbone.parameters())
        backbone_params = [p for p in self.parameters() if id(p) in backbone_ids]
        other_params    = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_scale},
            {"params": other_params,    "lr": base_lr},
        ]


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = DeepfakeDetector().to(device)
    dummy  = torch.randn(2, 30, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}  (expect: torch.Size([2]))")
    print(f"Output values: {out}")