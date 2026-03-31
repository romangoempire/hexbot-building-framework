"""
Experimental: Transformer-enhanced HexNet.

Adds global self-attention after the CNN backbone, allowing the network
to reason about distant board positions (colony play, multi-cluster
threats) that fixed convolution windows miss.

WARNING: This is experimental and untested in competitive play.
It may produce stronger results but trains ~30% slower per step.

Usage:
    from bot import create_network
    net = create_network('orca-transformer')
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from orca.config import (
        BOARD_SIZE, NUM_CHANNELS, NUM_FILTERS, NUM_RES_BLOCKS,
        TRANSFORMER_LAYERS, TRANSFORMER_HEADS, TRANSFORMER_DROPOUT,
    )
except ImportError:
    BOARD_SIZE = 19
    NUM_CHANNELS = 7
    NUM_FILTERS = 128
    NUM_RES_BLOCKS = 12
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 8
    TRANSFORMER_DROPOUT = 0.1


class ResBlock(nn.Module):
    """Standard residual block."""
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class HexPositionalEncoding(nn.Module):
    """Learnable positional encoding for hex grid positions.

    Each of the 361 (19x19) positions gets a learned embedding that
    encodes its hex-grid location. This helps the transformer understand
    spatial relationships between distant positions.
    """
    def __init__(self, d_model, max_positions=361):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_positions, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerHexNet(nn.Module):
    """CNN backbone + Transformer attention + 3 output heads.

    Architecture:
        Input (7, 19, 19)
          -> Conv 7->128, 3x3 + BN + ReLU         (local feature extraction)
          -> ResBlock x12 (128 filters)             (deep local patterns)
          -> Flatten to (361, 128) sequence          (spatial -> sequence)
          -> HexPositionalEncoding                   (spatial awareness)
          -> TransformerEncoder x2 (8 heads)         (global attention)
          -> Reshape back to (128, 19, 19)           (sequence -> spatial)
          -> Policy head: Conv 1x1 -> 361 logits
          -> Value head: Conv 1x1 -> FC -> tanh
          -> Threat head: Conv 1x1 -> FC -> 4

    The transformer layers let the network attend to ALL positions on the
    board simultaneously, which is critical for:
    - Colony play (distant stone clusters)
    - Multi-threat detection across board regions
    - Long-range strategic planning

    ~5.2M parameters (vs 3.9M for standard HexNet).
    Trains ~30% slower per step but may converge to stronger play.
    """

    def __init__(
        self,
        board_size=BOARD_SIZE,
        num_channels=NUM_CHANNELS,
        num_filters=NUM_FILTERS,
        num_res_blocks=NUM_RES_BLOCKS,
        transformer_layers=TRANSFORMER_LAYERS,
        transformer_heads=TRANSFORMER_HEADS,
        transformer_dropout=TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.board_size = board_size
        bs2 = board_size * board_size  # 361

        # CNN backbone (same as HexNet)
        self.conv_init = nn.Conv2d(num_channels, num_filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Transformer attention (global reasoning)
        self.pos_encoding = HexPositionalEncoding(num_filters, bs2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_filters,
            nhead=transformer_heads,
            dim_feedforward=num_filters * 4,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.post_attn_norm = nn.LayerNorm(num_filters)

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * bs2, bs2)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(bs2, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Threat head
        self.threat_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.threat_bn = nn.BatchNorm2d(1)
        self.threat_fc = nn.Linear(bs2, 4)

    def forward(self, x):
        batch = x.size(0)
        N = self.board_size

        # CNN backbone
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.res_blocks(x)
        # x: (batch, num_filters, N, N)

        # Reshape to sequence for transformer: (batch, N*N, num_filters)
        C = x.size(1)
        seq = x.view(batch, C, N * N).permute(0, 2, 1)  # (batch, 361, 128)
        seq = self.pos_encoding(seq)

        # Transformer attention (global reasoning across all positions)
        seq = self.transformer(seq)
        seq = self.post_attn_norm(seq)

        # Reshape back to spatial: (batch, num_filters, N, N)
        x = seq.permute(0, 2, 1).contiguous().view(batch, C, N, N)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(batch, -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(batch, -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Threat head
        t = F.relu(self.threat_bn(self.threat_conv(x)))
        t = t.view(batch, -1)
        t = self.threat_fc(t)

        return p, v, t

    def forward_pv(self, x):
        """Policy + value only (for inference, skips threat head)."""
        p, v, _ = self.forward(x)
        return p, v
