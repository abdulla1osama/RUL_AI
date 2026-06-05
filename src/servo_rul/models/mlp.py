import torch
from torch import nn


class MLPRUL(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], dropout: float = 0.2):
        super().__init__()

        layers: list[nn.Module] = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)