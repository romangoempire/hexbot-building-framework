"""
Hex-Masked Convolution for Connect-6.

Masks the 3x3 conv filter to only activate on hex neighbors, zeroing
the top-left and bottom-right corners that aren't hex-adjacent in
axial coordinates.

Standard 3x3 kernel:        Hex-masked kernel:
  [a] [b] [c]                [0] [b] [c]
  [d] [e] [f]    -->         [d] [e] [f]
  [g] [h] [i]                [g] [h] [0]

This keeps the spatial structure of CNN (fast, well-understood) while
respecting hex adjacency (no learning non-neighbor relationships).

Credit: Simon for the idea.

Usage:
    from bot import create_network
    net = create_network('hex-masked')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from orca.config import (
        BOARD_SIZE, NUM_CHANNELS, NUM_FILTERS, NUM_RES_BLOCKS,
    )
except ImportError:
    BOARD_SIZE = 19
    NUM_CHANNELS = 7
    NUM_FILTERS = 128
    NUM_RES_BLOCKS = 12


# Hex neighbor mask for 3x3 kernel in axial coordinates
# 0 = not a hex neighbor, 1 = hex neighbor (or self)
HEX_MASK_3x3 = torch.tensor([
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
], dtype=torch.float32)


class HexMaskedConv2d(nn.Module):
    """Conv2d with hex-neighbor masking on 3x3 kernels.

    Zeros out weights at positions (0,0) and (2,2) in every 3x3 filter,
    ensuring the convolution only aggregates from hex-adjacent cells.
    The mask is applied during forward pass so gradients never flow
    through non-neighbor positions.

    For non-3x3 kernels (e.g. 1x1), acts as a regular Conv2d.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=bias)
        self.kernel_size = kernel_size

        if kernel_size == 3:
            # Register mask as buffer (moves to device with model, not a parameter)
            mask = HEX_MASK_3x3.view(1, 1, 3, 3).expand(
                out_channels, in_channels, 3, 3).contiguous()
            self.register_buffer('mask', mask)
        else:
            self.mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            # Apply mask: zero out non-hex-neighbor weights
            masked_weight = self.conv.weight * self.mask
            return F.conv2d(x, masked_weight, self.conv.bias,
                            self.conv.stride, self.conv.padding)
        return self.conv(x)


class HexResBlock(nn.Module):
    """Residual block with hex-masked convolutions."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = HexMaskedConv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = HexMaskedConv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class HexMaskedNet(nn.Module):
    """HexNet with hex-masked convolutions.

    Identical architecture to standard HexNet but every 3x3 conv
    uses hex-masked filters. Same parameter count, same speed,
    but respects hex topology by preventing the network from
    learning relationships with non-hex-neighbors.

    Architecture:
        Input (7, 19, 19)
          -> HexMaskedConv 7->128, 3x3 + BN + ReLU
          -> HexResBlock x12 (128 filters, masked 3x3 convs)
          -> Policy head: Conv 1x1 -> FC -> 361
          -> Value head: Conv 1x1 -> FC -> 256 -> 1 (tanh)
          -> Threat head: Conv 1x1 -> FC -> 4

    ~3.9M parameters (same as standard HexNet).
    """

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        num_channels: int = NUM_CHANNELS,
        num_filters: int = NUM_FILTERS,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ):
        super().__init__()
        self.board_size = board_size
        bs2 = board_size * board_size

        # Initial block (hex-masked 3x3)
        self.conv_init = HexMaskedConv2d(num_channels, num_filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_filters)

        # Residual tower (hex-masked)
        self.res_blocks = nn.Sequential(
            *[HexResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head (1x1 conv, no masking needed)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.size(0)

        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(batch, -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(batch, -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Threat
        t = F.relu(self.threat_bn(self.threat_conv(x)))
        t = t.view(batch, -1)
        t = self.threat_fc(t)

        return p, v, t

    def forward_pv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p, v, _ = self.forward(x)
        return p, v
