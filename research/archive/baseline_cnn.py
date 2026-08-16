import torch
import torch.nn as nn
from torchvision import models


class BaselineDetector(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = models.convnext_tiny(weights=weights)

        # Количество признаков перед финальным слоем
        num_ftrs = self.model.classifier[2].in_features

        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # Оригинальный LayerNorm2d
            self.model.classifier[1],  # Оригинальный Flatten
            nn.Dropout(p=0.4),         # Dropout 40%
            nn.Linear(num_ftrs, 1)     # Бинарный выход (1 логит)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Алиас для совместимости
BaselineCNN = BaselineDetector


if __name__ == "__main__":
    model = BaselineDetector(pretrained=True)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print("Архитектура инициализирована успешно.")
    print(f"Размерность выхода модели: {output.shape}")  # torch.Size([4, 1])