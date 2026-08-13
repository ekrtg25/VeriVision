import torch
import torch.nn as nn
from torchvision import models

class BaselineDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = models.convnext_tiny(weights=weights)
        
        # Получаем количество признаков перед финальным слоем
        num_ftrs = self.model.classifier[2].in_features
        
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # Сохраняем оригинальный LayerNorm2d
            self.model.classifier[1],  # Сохраняем Flatten
            nn.Dropout(p=0.4),         # Новый слой: Размытие фокуса (Dropout 40%)
            nn.Linear(num_ftrs, 1)     # Бинарный выход (Fake/Real)
        )

    def forward(self, x):
        return self.model(x)

# Проверка работоспособности архитектуры при прямом запуске скрипта
if __name__ == "__main__":
    # Инициализируем модель (убран лишний аргумент)
    model = BaselineDetector(pretrained=True)
    dummy_input = torch.randn(4, 3, 224, 224) # Batch size 4, 3 channels, 224x224
    output = model(dummy_input)
    
    print("Архитектура инициализирована успешно.")
    print(f"Размерность выхода модели: {output.shape}") 
    # Ожидается: torch.Size([4, 1])