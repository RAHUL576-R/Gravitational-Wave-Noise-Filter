
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Pre-activation residual block with BatchNorm."""
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.BatchNorm1d(ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Learns to re-weight each channel (≈ frequency band) independently.
    """
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x).unsqueeze(-1)   # (B, C, 1)
        return x * w


class DilatedBottleneck(nn.Module):
    """
    Multi-scale dilated convolution block.
    Rates 1, 2, 4, 8 are concatenated then projected back to `ch` channels.
    Gives a large temporal receptive field without extra downsampling.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(ch, ch // 4, 3, padding=r, dilation=r),
                nn.BatchNorm1d(ch // 4),
                nn.LeakyReLU(0.1, inplace=True),
            )
            for r in [1, 2, 4, 8]
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(ch, ch, 1),
            nn.BatchNorm1d(ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.attn = ChannelAttention(ch)
        self.res  = ResBlock(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([b(x) for b in self.branches], dim=1)  # (B, ch, L)
        out = self.proj(out)
        out = self.attn(out)
        out = self.res(out)
        return out + x     # global skip


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder stages
# ─────────────────────────────────────────────────────────────────────────────

class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(out_ch),
            ResBlock(out_ch),
        )

    def forward(self, x):
        return self.net(x)


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # after concat: in_ch + skip_ch
        self.up   = nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)
        self.net  = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            ResBlock(out_ch),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad/trim to match skip length (handles odd-length mismatches)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full WaveUNet
# ─────────────────────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    
    # Channel widths at each encoder depth
    CHANNELS = [32, 64, 128, 256]

    def __init__(self):
        super().__init__()

        ch = self.CHANNELS

        # Input projection
        self.in_conv = nn.Sequential(
            nn.Conv1d(1, ch[0], 3, padding=1),
            nn.BatchNorm1d(ch[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Encoder
        self.enc1 = EncoderStage(ch[0], ch[0])   # 4096 → 2048
        self.enc2 = EncoderStage(ch[0], ch[1])   # 2048 → 1024
        self.enc3 = EncoderStage(ch[1], ch[2])   # 1024 →  512
        self.enc4 = EncoderStage(ch[2], ch[3])   #  512 →  256

        # Bottleneck
        self.bottleneck = DilatedBottleneck(ch[3])

        # Decoder  (in_ch, skip_ch, out_ch)
        self.dec4 = DecoderStage(ch[3], ch[2], ch[2])   # 256  → 512
        self.dec3 = DecoderStage(ch[2], ch[1], ch[1])   # 512  → 1024
        self.dec2 = DecoderStage(ch[1], ch[0], ch[0])   # 1024 → 2048
        self.dec1 = DecoderStage(ch[0], ch[0], ch[0])   # 2048 → 4096

        # Output projection — NO activation: output is unbounded
        self.out_conv = nn.Conv1d(ch[0], 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, a=0.1,
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection — keep at full resolution for skip
        x0 = self.in_conv(x)       # (B, 32, 4096)

        # Encoder
        e1 = self.enc1(x0)         # (B,  32, 2048)
        e2 = self.enc2(e1)         # (B,  64, 1024)
        e3 = self.enc3(e2)         # (B, 128,  512)
        e4 = self.enc4(e3)         # (B, 256,  256)

        # Bottleneck
        b  = self.bottleneck(e4)   # (B, 256,  256)

        # Decoder with skip connections
        d4 = self.dec4(b,  e3)     # (B, 128,  512)
        d3 = self.dec3(d4, e2)     # (B,  64, 1024)
        d2 = self.dec2(d3, e1)     # (B,  32, 2048)
        d1 = self.dec1(d2, x0)     # (B,  32, 4096)

        return self.out_conv(d1)   # (B,   1, 4096)


if __name__ == "__main__":
    # Quick sanity check
    m   = Autoencoder()
    x   = torch.randn(4, 1, 4096)
    out = m(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"✅  Output shape: {out.shape}   Parameters: {params:,}")
