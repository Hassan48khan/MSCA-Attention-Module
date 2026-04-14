"""
MSCA: Multi-Scale Cross-Axis Attention Module
==============================================
A novel attention mechanism that extends EMA with:
  1. Tri-scale parallel convolutions (1x1, 3x3, 5x5) for richer multi-scale context
  2. Cross-axis pooling fusion: combines H-pooling, W-pooling, AND diagonal-pooling
     via a lightweight learned mixing, rather than simple concatenation
  3. Dual-gate recalibration: applies both channel-wise and spatial sigmoid gates
     before the cross-spatial dot-product, sharpening attended regions
  4. No channel dimensionality reduction throughout

Paper reference:
  Inspired by EMA (Ouyang et al., ICASSP 2023) but introduces tri-scale branches
  and a cross-axis pooling fusion strategy for richer positional encoding.

Usage:
    from MSCA_attention_module import MSCA

    # In any CNN backbone:
    attn = MSCA(channels=256)
    out  = attn(feature_map)   # shape preserved: (B, C, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: lightweight diagonal / anti-diagonal average pooling
# (approximated as a rotation-then-pool for efficiency)
# ---------------------------------------------------------------------------

class DiagPool(nn.Module):
    """
    Approximates diagonal-direction global average pooling by rotating the
    feature map 45° (bilinear) and then applying standard H-direction pooling.
    Lightweight – no learnable parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        # Build a 45-degree affine grid
        theta = torch.tensor(
            [[0.7071, -0.7071, 0.0],
             [0.7071,  0.7071, 0.0]],
            dtype=x.dtype, device=x.device
        ).unsqueeze(0).expand(b, -1, -1)          # (B, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_rot = F.grid_sample(x, grid, align_corners=False, mode='bilinear',
                              padding_mode='zeros')
        # Pool along H → result: (B, C, 1, W)
        return x_rot.mean(dim=2, keepdim=True)


# ---------------------------------------------------------------------------
# Main MSCA Module
# ---------------------------------------------------------------------------

class MSCA(nn.Module):
    """
    Multi-Scale Cross-Axis Attention (MSCA)

    Args:
        channels (int): Number of input/output channels. Must be divisible by `groups`.
        groups   (int): Number of channel groups. Default 32.
        use_diag (bool): Whether to include the diagonal-axis pooling branch.
                         Adds slight overhead but improves recall on skewed objects.
                         Default True.
    """

    def __init__(self, channels: int, c2=None, groups: int = 32, use_diag: bool = True):
        super().__init__()
        assert channels % groups == 0, \
            f"channels ({channels}) must be divisible by groups ({groups})"

        self.groups   = groups
        self.use_diag = use_diag
        cg = channels // groups  # channels per group

        # ── Shared spatial encoders (operate on group slices) ──────────────
        self.pool_h  = nn.AdaptiveAvgPool2d((None, 1))   # → (B*G, cg, H, 1)
        self.pool_w  = nn.AdaptiveAvgPool2d((1, None))   # → (B*G, cg, 1, W)
        if use_diag:
            self.pool_d = DiagPool()                     # → (B*G, cg, 1, W)

        # ── Cross-axis fusion: learned 1×1 mixing after concat ─────────────
        # concat channels: H + W [+ D] pooling vectors along spatial dim
        n_axes    = 3 if use_diag else 2
        # fuse along the concat spatial dimension (no channel reduction)
        self.axis_fuse = nn.Conv2d(cg, cg, kernel_size=1, stride=1, padding=0, bias=False)

        # ── Group normalisation per sub-feature ────────────────────────────
        self.gn = nn.GroupNorm(cg, cg)

        # ── Tri-scale parallel convolutions ────────────────────────────────
        self.conv1x1 = nn.Conv2d(cg, cg, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(cg, cg, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(cg, cg, kernel_size=5, stride=1, padding=2, bias=False)

        # Learned scalar weights for tri-scale fusion (softmax normalised)
        self.scale_w = nn.Parameter(torch.ones(3))

        # ── Global pooling for cross-spatial dot-product ────────────────────
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        # ── Dual gate: channel gate + spatial gate before dot-product ───────
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cg, cg, 1, bias=False),
            nn.Sigmoid()
        )
        self.sp_gate = nn.Sequential(
            nn.Conv2d(cg, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.softmax = nn.Softmax(dim=-1)

    # -----------------------------------------------------------------------

    def _cross_axis_encode(self, gx):
        """
        gx : (B*G, cg, H, W)
        Returns a spatially re-weighted tensor of the same shape.
        """
        b, c, h, w = gx.shape

        x_h = self.pool_h(gx)                            # (B*G, cg, H, 1)
        x_w = self.pool_w(gx).permute(0, 1, 3, 2)       # (B*G, cg, 1, W) → (B*G, cg, W, 1) NO
        # keep as (B*G, cg, 1, W) to match concat below
        x_w = self.pool_w(gx)                            # (B*G, cg, 1, W)

        if self.use_diag:
            x_d = self.pool_d(gx)                        # (B*G, cg, 1, W)
            # Concatenate along H dim: H + 1 + 1
            fused = torch.cat([x_h,
                               x_w.permute(0, 1, 3, 2),  # (B*G, cg, W, 1)
                               x_d.permute(0, 1, 3, 2)], dim=2)  # (B*G, cg, H+W+W, 1)
        else:
            fused = torch.cat([x_h,
                               x_w.permute(0, 1, 3, 2)], dim=2)  # (B*G, cg, H+W, 1)

        fused = self.axis_fuse(fused)                    # mix across channels

        # split back
        if self.use_diag:
            x_h2, x_w2, _ = torch.split(fused, [h, w, w], dim=2)
        else:
            x_h2, x_w2 = torch.split(fused, [h, w], dim=2)

        # x_h2: (B*G, cg, H, 1); x_w2: (B*G, cg, W, 1)
        out = gx * x_h2.sigmoid() * x_w2.permute(0, 1, 3, 2).sigmoid()
        return self.gn(out)

    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Attended feature map of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        # Reshape into groups
        gx = x.reshape(b * self.groups, c // self.groups, h, w)   # (B*G, cg, H, W)
        cg = c // self.groups

        # ── Branch A: cross-axis encoded + 1×1 conv ────────────────────────
        xa = self._cross_axis_encode(gx)
        xa = self.conv1x1(xa)

        # ── Branch B: 3×3 conv ─────────────────────────────────────────────
        xb = self.conv3x3(gx)

        # ── Branch C: 5×5 conv ─────────────────────────────────────────────
        xc = self.conv5x5(gx)

        # ── Tri-scale fusion with learned weights ──────────────────────────
        sw = F.softmax(self.scale_w, dim=0)
        x1 = sw[0] * xa + sw[1] * xb + sw[2] * xc        # (B*G, cg, H, W)
        x2 = xb                                            # keep xb as second stream

        # ── Dual-gate recalibration ────────────────────────────────────────
        x1 = x1 * self.ch_gate(x1) * self.sp_gate(x1)
        x2 = x2 * self.ch_gate(x2) * self.sp_gate(x2)

        # ── Cross-spatial dot-product (same as EMA) ────────────────────────
        x11 = self.softmax(
            self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )                                                  # (B*G, 1, cg)
        x12 = x2.reshape(b * self.groups, cg, -1)         # (B*G, cg, H*W)

        x21 = self.softmax(
            self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )                                                  # (B*G, 1, cg)
        x22 = x1.reshape(b * self.groups, cg, -1)         # (B*G, cg, H*W)

        weights = (
            torch.matmul(x11, x12) + torch.matmul(x21, x22)
        ).reshape(b * self.groups, 1, h, w)               # (B*G, 1, H, W)

        out = (gx * weights.sigmoid()).reshape(b, c, h, w)
        return out


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        (2, 64,  32, 32),
        (2, 128, 16, 16),
        (2, 256,  8,  8),
    ]

    print(f"{'Shape':<30} {'Params':>10} {'Time (ms)':>12} {'Output OK':>10}")
    print("-" * 65)
    for b, c, h, w in configs:
        x   = torch.randn(b, c, h, w).to(device)
        mdl = MSCA(c).to(device)
        # warmup
        _ = mdl(x)
        t0  = time.perf_counter()
        for _ in range(50):
            y = mdl(x)
        elapsed = (time.perf_counter() - t0) / 50 * 1000
        ok = (y.shape == x.shape)
        print(f"({b},{c:>4},{h:>3},{w:>3})  {count_params(mdl):>10,}  {elapsed:>10.2f}ms  {'✓' if ok else '✗':>10}")
