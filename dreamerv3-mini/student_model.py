"""
Lightweight Student models for behaviour cloning.

Supports two modalities:
  - Proprioceptive (MLP encoder) + continuous actions  (e.g. dmc_walker_walk)
  - Image (CNN encoder)          + discrete  actions  (e.g. crafter_reward)

Architecture (image path)
=========================
Input (64,64,3 uint8)
  -> /255  ->  permute to CHW
  -> Conv2d(3->32, k3, s2) + SiLU               # 32x32
  -> Conv2d(32->64, k3, s2) + SiLU              # 16x16
  -> Conv2d(64->64, k3, s2) + SiLU              # 8x8
  -> flatten (4096)
  -> Linear(4096 -> hidden) + SiLU
  -> [ResBlock] x N
  -> LayerNorm -> Linear(hidden -> act_dim)       # logits (discrete)

Architecture (proprio path)
===========================
Input (obs_dim float32)
  -> FixedNormalize(mean, std)
  -> Linear(obs_dim -> hidden) + SiLU
  -> [ResBlock] x N
  -> LayerNorm -> Linear(hidden -> act_dim) -> Tanh   # [-1, 1] (continuous)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class StudentConfig:
    # Observation
    obs_type:   str = 'proprio'          # 'proprio' | 'image'
    obs_dim:    int = 24                 # only for proprio

    # Action
    act_type:   str = 'continuous'       # 'continuous' | 'discrete'
    act_dim:    int = 6                  # continuous: output dim, discrete: num classes

    # Architecture
    encoder_type: str = 'mlp'            # 'mlp' | 'cnn'
    hidden:     int = 512
    blocks:     int = 4
    dropout:    float = 0.1

    # CNN-specific
    cnn_channels: Optional[Tuple[int, ...]] = None   # e.g. (32, 64, 64)
    cnn_kernel:   int = 3
    img_channels: int = 3                             # 1 for grayscale, 3 for RGB


class ResBlock(nn.Module):
    """Pre-norm residual block:
       LayerNorm -> Linear -> SiLU -> Dropout -> Linear -> residual add.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ImageEncoder(nn.Module):
    """Lightweight CNN: (B,H,W,C) uint8 -> (B, out_dim) float32."""

    def __init__(self, channels: Tuple[int, ...] = (32, 64, 64),
                 kernel: int = 3, in_channels: int = 3):
        super().__init__()
        layers = []
        in_ch = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel, stride=2,
                          padding=kernel // 2),
                nn.SiLU(),
            ])
            in_ch = out_ch
        self.convnet = nn.Sequential(*layers)
        # After 3x stride-2 on 64x64: 64->32->16->8
        self.out_dim = channels[-1] * 8 * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) uint8
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)        # (B, C, H, W)
        x = self.convnet(x)
        return x.flatten(1)               # (B, out_dim)


class StudentPolicy(nn.Module):
    """Unified student policy for both proprio-MLP and image-CNN encoders,
    with continuous (Tanh) or discrete (logits) output heads."""

    def __init__(
        self,
        config: StudentConfig,
        obs_mean: Optional[torch.Tensor] = None,
        obs_std:  Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config

        # ── Encoder ──────────────────────────────────────────────────────
        if config.encoder_type == 'mlp':
            assert obs_mean is not None and obs_std is not None
            self.register_buffer('obs_mean', obs_mean.float())
            self.register_buffer('obs_std',  obs_std.float())
            self.encoder = None                     # inline in forward
            self.input_proj = nn.Linear(config.obs_dim, config.hidden)
        else:  # 'cnn'
            channels = config.cnn_channels or (32, 64, 64)
            self.encoder = ImageEncoder(channels, config.cnn_kernel,
                                        config.img_channels)
            self.input_proj = nn.Linear(self.encoder.out_dim, config.hidden)

        # ── Residual backbone ────────────────────────────────────────────
        self.blocks = nn.Sequential(
            *[ResBlock(config.hidden, config.dropout)
              for _ in range(config.blocks)]
        )

        # ── Output head ─────────────────────────────────────────────────
        self.output_norm = nn.LayerNorm(config.hidden)
        self.output_proj = nn.Linear(config.hidden, config.act_dim)

        # Small init for continuous (match DreamerV3 outscale=0.01)
        if config.act_type == 'continuous':
            nn.init.uniform_(self.output_proj.weight, -0.01, 0.01)
            nn.init.zeros_(self.output_proj.bias)

        # ── Log-std head for NLL ablation (continuous only) ──────────────
        if config.act_type == 'continuous':
            self.log_std_proj = nn.Linear(config.hidden, config.act_dim)
            nn.init.zeros_(self.log_std_proj.weight)
            nn.init.constant_(self.log_std_proj.bias, -1.0)  # initial σ ≈ 0.37

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: obs → hidden features (after LayerNorm)."""
        if self.config.encoder_type == 'mlp':
            x = (x - self.obs_mean) / self.obs_std
            x = self.input_proj(x)
        else:
            x = self.encoder(x)
            x = self.input_proj(x)
        x = nn.functional.silu(x)
        x = self.blocks(x)
        return self.output_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encode(x)
        x = self.output_proj(h)
        if self.config.act_type == 'continuous':
            x = torch.tanh(x)
        return x

    def forward_with_log_std(self, x: torch.Tensor):
        """Return (mean_action, log_std) for Gaussian NLL loss.
        Only valid for continuous action spaces."""
        h = self._encode(x)
        mean = torch.tanh(self.output_proj(h))
        log_std = torch.clamp(self.log_std_proj(h), -4.0, 0.0)
        return mean, log_std


# Backward-compatible alias
StudentResMLP = StudentPolicy
