import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import models

class BaselineDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = models.convnext_tiny(weights=weights)
        
        # Меняем финальный слой классификатора под бинарный вывод
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)

# Проверка работоспособности архитектуры при прямом запуске скрипта
if __name__ == "__main__":
    model = BaselineDetector(num_classes=1, pretrained=True)
    dummy_input = torch.randn(4, 3, 224, 224) # Batch size 4, 3 channels, 224x224
    output = model(dummy_input)
    
    print(f"Архитектура инициализирована успешно.")
    print(f"Размерность выхода модели: {output.shape}") 
    # Ожидается: torch.Size([4, 1])