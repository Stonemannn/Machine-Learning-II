from typing import Any
import torch
from torch import Tensor
from torch import nn

__all__ = [
    "AlexNet",
    "alexnet",
]

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        print('Running AlexNet')
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (3, 3), (1, 1), (2, 2)),
            nn.ReLU(True),
            # nn.MaxPool2d((3, 3), (2, 2)),

            # nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            # nn.ReLU(True),
            # nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            # nn.ReLU(True),
            # nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            # nn.ReLU(True),
            # nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1728, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def alexnet(**kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    return model