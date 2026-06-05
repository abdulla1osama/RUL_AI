import torch
from torch import nn


class ResBlock1D(nn.Module):
    """Conv → BN → ReLU → Conv → BN + residual"""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class MultiScaleBlock(nn.Module):
    """Parallel convolutions at kernel sizes 3, 5, 7 — concatenated"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        # out_channels must be divisible by 3
        branch = out_channels // 3
        self.k3 = nn.Sequential(
            nn.Conv1d(in_channels, branch, 3, padding=1, bias=False),
            nn.BatchNorm1d(branch), nn.ReLU(),
        )
        self.k5 = nn.Sequential(
            nn.Conv1d(in_channels, branch, 5, padding=2, bias=False),
            nn.BatchNorm1d(branch), nn.ReLU(),
        )
        self.k7 = nn.Sequential(
            nn.Conv1d(in_channels, branch, 7, padding=3, bias=False),
            nn.BatchNorm1d(branch), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(torch.cat([self.k3(x), self.k5(x), self.k7(x)], dim=1))


class CNN1DRUL(nn.Module):
    def __init__(
        self,
        num_features: int,
        channels: list[int] = [64, 128, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ── Stem: project raw features into channel space ──────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(num_features, channels[0], kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
        )

        # ── Multi-scale block: captures short and long temporal patterns ────
        # channels[0] → channels[1], must be divisible by 3
        ms_out = (channels[1] // 3) * 3  # round down to nearest multiple of 3
        self.multiscale = MultiScaleBlock(channels[0], ms_out, dropout)

        # ── Residual blocks for deeper representation ────────────────────────
        self.res1 = ResBlock1D(ms_out, kernel_size, dropout)
        self.res2 = ResBlock1D(ms_out, kernel_size, dropout)

        # ── Channel expansion ────────────────────────────────────────────────
        self.expand = nn.Sequential(
            nn.Conv1d(ms_out, channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(),
        )
        self.res3 = ResBlock1D(channels[-1], kernel_size, dropout)

        # ── Pooling head: both avg and max, then regress ─────────────────────
        self.head = nn.Sequential(
            nn.Linear(channels[-1] * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, features]
        x = x.transpose(1, 2)           # → [batch, features, window]

        x = self.stem(x)
        x = self.multiscale(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.expand(x)
        x = self.res3(x)

        # Both avg and max pooling — avg captures trend, max captures worst point
        avg = x.mean(dim=-1)
        mx  = x.max(dim=-1).values
        x   = torch.cat([avg, mx], dim=1)  # → [batch, channels[-1]*2]

        return self.head(x).squeeze(-1)    # → [batch]