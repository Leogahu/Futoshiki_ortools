import torch
import torch.nn as nn


class SignCNN(nn.Module):
    """
    CNN para clasificar signos de desigualdad entre celdas.
    Clases: 0=nada, 1="<", 2=">".
    Acepta cualquier tamano de entrada >= 48x48 gracias al AdaptiveAvgPool.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # fuerza salida (64, 6, 8) sin importar el tamano de entrada
            nn.AdaptiveAvgPool2d((6, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))