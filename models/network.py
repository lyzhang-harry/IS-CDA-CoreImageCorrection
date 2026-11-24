import torch
from torch import nn
from torch.nn import functional as F


def brightness_head(channels):
    head = nn.Conv1d(channels, 2, 1)
    # Preserve the initialization sequence used for pixel sampling.
    for _ in range(2):
        nn.Conv1d(channels, 2, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


class ColumnBackbone(nn.Module):
    def __init__(self, F, C=48):
        super().__init__()
        self.conv1 = nn.Conv1d(F, C, 5, padding=2)
        self.conv2 = nn.Conv1d(C, C, 5, padding=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))))


class ColorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(8, 48, 5, padding=2, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv1d(48, 48, 5, padding=2, padding_mode="replicate"),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(48, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, gate):
        small = F.interpolate(features, size=32, mode="linear", align_corners=True)
        raw = self.head(self.backbone(small))
        full = F.interpolate(
            raw.unsqueeze(2), size=(1, features.shape[-1]), mode="bicubic", align_corners=True
        )[0, :, 0].T
        bounded = torch.tanh(full) * gate[:, None]
        fields = torch.stack(
            [
                torch.exp(0.7 * bounded[:, 0]),
                (10 / 360) * bounded[:, 1],
                32 * bounded[:, 2],
                32 * bounded[:, 3],
            ],
            1,
        )
        return fields, bounded
