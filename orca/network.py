"""
orca/network.py - Neural network architectures and utilities.

Contains:
- ResBlock, SEResBlock (residual blocks)
- HexNet (standard CNN)
- HybridHexNet (CNN + attention)
- create_network factory
- OnnxPredictor, export_onnx
- Checkpoint migration helpers
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import config constants
try:
    from orca.config import (
        BOARD_SIZE, NUM_CHANNELS, NUM_FILTERS, NUM_RES_BLOCKS,
    )
except ImportError:
    BOARD_SIZE = 19
    NUM_CHANNELS = 7
    NUM_FILTERS = 128
    NUM_RES_BLOCKS = 12


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation (channel attention)."""

    def __init__(self, num_filters: int, se_ratio: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        # SE: global avg pool -> FC -> ReLU -> FC -> sigmoid
        mid = max(num_filters // se_ratio, 16)
        self.se_fc1 = nn.Linear(num_filters, mid)
        self.se_fc2 = nn.Linear(mid, num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # SE attention
        se = out.mean(dim=(2, 3))  # (B, C)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se.unsqueeze(-1).unsqueeze(-1)
        return F.relu(out + residual)


# ---------------------------------------------------------------------------
# Checkpoint migration
# ---------------------------------------------------------------------------

def migrate_checkpoint_5to7(state_dict: dict) -> dict:
    """Migrate a 5-channel checkpoint to 7-channel architecture.
    Preserves all learned weights; new channels initialized to zero."""
    key = 'conv_init.weight'
    if key in state_dict:
        w = state_dict[key]
        if w.shape[1] == 5:
            nf = w.shape[0]
            new_w = torch.zeros(nf, 7, 3, 3, dtype=w.dtype, device=w.device)
            new_w[:, :5, :, :] = w
            state_dict[key] = new_w
            print(f'  OK: Migrated conv_init.weight: {w.shape} -> {new_w.shape} (2 threat channels added)')
    return state_dict


def migrate_checkpoint_filters(state_dict: dict, target_filters: int = NUM_FILTERS,
                                target_blocks: int = NUM_RES_BLOCKS) -> dict:
    """Migrate a smaller network checkpoint to a larger one.
    Pads filter dimensions with small random values. Copies existing blocks,
    initializes new blocks randomly."""
    migrated = False
    new_sd = {}

    for key, tensor in state_dict.items():
        new_sd[key] = tensor  # default: copy as-is

    # Check if migration is needed by looking at conv_init
    init_key = 'conv_init.weight'
    if init_key in new_sd:
        old_filters = new_sd[init_key].shape[0]
        if old_filters < target_filters:
            print(f'  OK: Migrating {old_filters}->{target_filters} filters, expanding network...')
            migrated = True

            # Helper: pad a tensor's filter dimensions
            def pad_filters(t, dim0=None, dim1=None):
                if t.dim() == 4:  # Conv weight: (out, in, H, W)
                    out_f = dim0 or t.shape[0]
                    in_f = dim1 or t.shape[1]
                    new_t = torch.randn(out_f, in_f, t.shape[2], t.shape[3],
                                        dtype=t.dtype) * 0.01
                    new_t[:t.shape[0], :t.shape[1], :, :] = t
                    return new_t
                elif t.dim() == 1:  # BN weight/bias/running_mean/running_var
                    new_t = torch.zeros(dim0 or t.shape[0], dtype=t.dtype)
                    new_t[:t.shape[0]] = t
                    if 'weight' in key:  # BN weight should be 1, not 0
                        new_t[t.shape[0]:] = 1.0
                    return new_t
                elif t.dim() == 2:  # Linear weight: (out, in)
                    out_f = dim0 or t.shape[0]
                    in_f = dim1 or t.shape[1]
                    new_t = torch.randn(out_f, in_f, dtype=t.dtype) * 0.01
                    new_t[:t.shape[0], :t.shape[1]] = t
                    return new_t
                return t

            nf = target_filters
            nc = new_sd[init_key].shape[1]  # input channels (7)
            bs2 = BOARD_SIZE * BOARD_SIZE

            # conv_init: (old_f, 7, 3, 3) -> (nf, 7, 3, 3)
            new_sd[init_key] = pad_filters(new_sd[init_key], nf, nc)
            new_sd['bn_init.weight'] = pad_filters(new_sd['bn_init.weight'], nf)
            new_sd['bn_init.bias'] = pad_filters(new_sd['bn_init.bias'], nf)
            new_sd['bn_init.running_mean'] = pad_filters(new_sd['bn_init.running_mean'], nf)
            new_sd['bn_init.running_var'] = pad_filters(new_sd.get('bn_init.running_var',
                                                        torch.ones(old_filters)), nf)
            if 'bn_init.num_batches_tracked' in new_sd:
                pass  # scalar, no migration needed

            # Res blocks: migrate existing, leave new ones for random init
            for i in range(target_blocks):
                prefix = f'res_blocks.{i}'
                for suffix in ['.conv1.weight', '.bn1.weight', '.bn1.bias',
                               '.bn1.running_mean', '.bn1.running_var',
                               '.conv2.weight', '.bn2.weight', '.bn2.bias',
                               '.bn2.running_mean', '.bn2.running_var']:
                    k = prefix + suffix
                    if k in new_sd:
                        t = new_sd[k]
                        if 'conv' in suffix and t.dim() == 4:
                            new_sd[k] = pad_filters(t, nf, nf)
                        elif t.dim() == 1:
                            new_sd[k] = pad_filters(t, nf)
                    # If key doesn't exist (new block), it'll be randomly initialized by the model

            # Policy head: conv (nf->2), fc (2*bs2 -> bs2)
            if 'policy_conv.weight' in new_sd:
                new_sd['policy_conv.weight'] = pad_filters(
                    new_sd['policy_conv.weight'], 2, nf)
            # policy_fc stays same size (2*bs2 -> bs2)

            # Value head: conv (nf->1), fc1 (bs2->256), fc2 (256->1)
            if 'value_conv.weight' in new_sd:
                new_sd['value_conv.weight'] = pad_filters(
                    new_sd['value_conv.weight'], 1, nf)

            # Threat head: conv (nf->1)
            if 'threat_conv.weight' in new_sd:
                new_sd['threat_conv.weight'] = pad_filters(
                    new_sd['threat_conv.weight'], 1, nf)

            print(f'  OK: Network expanded: {old_filters}->{nf} filters')

    return new_sd


# ---------------------------------------------------------------------------
# HexNet
# ---------------------------------------------------------------------------

class HexNet(nn.Module):
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

        # Initial block
        self.conv_init = nn.Conv2d(num_channels, num_filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * bs2, bs2)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(bs2, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Threat detection head: predicts [my_4, my_5, opp_4, opp_5] in a row
        self.threat_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.threat_bn = nn.BatchNorm2d(1)
        self.threat_fc = nn.Linear(bs2, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Initial block
        x = F.relu(self.bn_init(self.conv_init(x)))
        # Residual tower
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # raw logits (batch, BS*BS)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # (batch, 1)

        # Threat head
        t = F.relu(self.threat_bn(self.threat_conv(x)))
        t = t.view(t.size(0), -1)
        t = self.threat_fc(t)  # (batch, 4)

        return p, v, t

    def forward_pv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Policy + value only (for ONNX export and inference)."""
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.res_blocks(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

    @torch.no_grad()
    def predict(
        self, encoded_state: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Single-state inference. Returns (policy_logits, value_scalar)."""
        self.eval()
        x = encoded_state.unsqueeze(0)
        if next(self.parameters()).device.type != x.device.type:
            x = x.to(next(self.parameters()).device)
        p, v = self.forward_pv(x)
        return p.squeeze(0).cpu(), v.item()


# ---------------------------------------------------------------------------
# HybridHexNet (CNN + Global Attention)
# ---------------------------------------------------------------------------

class HybridHexNet(nn.Module):
    """CNN + Global Attention network for hex connect-6.

    Architecture:
      1. Stem: Conv2d -> BN -> ReLU
      2. Body: N x SE-ResBlock (local patterns + channel attention)
      3. Global Attention: 2 x MultiheadAttention (long-range dependencies)
      4. Heads: policy (361), value (1), threat (4)
    """

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        num_channels: int = NUM_CHANNELS,
        num_filters: int = 256,
        num_res_blocks: int = 12,
        num_attention_heads: int = 8,
        num_attention_layers: int = 2,
    ):
        super().__init__()
        self.board_size = board_size
        bs2 = board_size * board_size

        # Stem
        self.conv_init = nn.Conv2d(num_channels, num_filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_filters)

        # SE-ResBlock tower
        self.res_blocks = nn.Sequential(
            *[SEResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Global attention layers
        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        for _ in range(num_attention_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(num_filters, num_attention_heads, batch_first=True)
            )
            self.attn_norms.append(nn.LayerNorm(num_filters))

        # Hex positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, bs2, num_filters) * 0.02)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.size(0)
        bs = self.board_size
        bs2 = bs * bs

        # Stem + ResBlocks
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.res_blocks(x)

        # Global attention: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.view(B, -1, bs2).permute(0, 2, 1)  # (B, bs2, C)
        x_flat = x_flat + self.pos_embedding

        for attn, norm in zip(self.attn_layers, self.attn_norms):
            residual = x_flat
            x_flat = norm(x_flat)
            attn_out, _ = attn(x_flat, x_flat, x_flat)
            x_flat = residual + attn_out

        # Back to spatial: (B, bs2, C) -> (B, C, H, W)
        x = x_flat.permute(0, 2, 1).view(B, -1, bs, bs)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(B, -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(B, -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Threat head
        t = F.relu(self.threat_bn(self.threat_conv(x)))
        t = t.view(B, -1)
        t = self.threat_fc(t)

        return p, v, t

    def forward_pv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p, v, _ = self.forward(x)
        return p, v

    @torch.no_grad()
    def predict(self, encoded_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        self.eval()
        x = encoded_state.unsqueeze(0)
        dev = next(self.parameters()).device
        if x.device != dev:
            x = x.to(dev)
        p, v = self.forward_pv(x)
        return p.squeeze(0).cpu(), v.item()


# ---------------------------------------------------------------------------
# ONNX support
# ---------------------------------------------------------------------------

class OnnxPredictor:
    """Drop-in replacement for HexNet.predict() using ONNX Runtime."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=opts,
            providers=['CPUExecutionProvider']
        )

    def predict(self, encoded_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        inp = {'state': encoded_state.unsqueeze(0).numpy()}
        policy, value = self.sess.run(None, inp)
        return torch.from_numpy(policy[0]), float(value[0][0])


def export_onnx(net: HexNet, path: str = '/tmp/hex_model.onnx'):
    """Export the policy+value heads to ONNX for fast CPU inference."""
    import warnings
    net_cpu = net.cpu()
    net_cpu.eval()
    dummy = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # Use a wrapper that calls forward_pv
    class _PVWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            return self.net.forward_pv(x)

    wrapper = _PVWrapper(net_cpu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper, dummy, path,
            input_names=['state'],
            output_names=['policy', 'value'],
        )
    return path


# ---------------------------------------------------------------------------
# Network factory
# ---------------------------------------------------------------------------

def create_network(config: str = 'standard', board_size: int = BOARD_SIZE) -> nn.Module:
    """Factory for network architectures."""
    if config == 'fast':
        return HexNet(board_size=board_size, num_filters=64, num_res_blocks=4)
    elif config == 'standard':
        return HexNet(board_size=board_size, num_filters=128, num_res_blocks=12)
    elif config == 'large':
        return HexNet(board_size=board_size, num_filters=256, num_res_blocks=12)
    elif config == 'hybrid':
        return HybridHexNet(board_size=board_size, num_filters=256, num_res_blocks=12)
    elif config == 'hybrid-small':
        return HybridHexNet(board_size=board_size, num_filters=128, num_res_blocks=6,
                            num_attention_heads=4, num_attention_layers=1)
    elif config == 'orca-transformer':
        from orca.transformer_net import TransformerHexNet
        return TransformerHexNet(board_size=board_size)
    elif config == 'hex-masked':
        from orca.hex_conv import HexMaskedNet
        return HexMaskedNet(board_size=board_size)
    elif config == 'hex-gnn':
        from orca.hex_gnn import HexGNN
        return HexGNN(board_size=board_size)
    elif config == 'multiscale':
        from orca.multiscale_net import MultiScaleHexNet
        return MultiScaleHexNet(board_size=board_size)
    else:
        raise ValueError(f"Unknown network config: {config}. "
                         f"Available: fast, standard, large, hybrid, hybrid-small, "
                         f"orca-transformer, hex-gnn, multiscale")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'ResBlock', 'SEResBlock',
    'HexNet', 'HybridHexNet',
    'OnnxPredictor', 'export_onnx',
    'create_network',
    'migrate_checkpoint_5to7', 'migrate_checkpoint_filters',
]
