import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class BaselineDetector(nn.Module):
    """
    Базовая модель для бинарной детекции AI-сгенерированных изображений.
    Использует предобученный ResNet50 с замененным классификатором.
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Загружаем веса, если нужно
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Опциональная заморозка feature extractor'а (полезно для быстрого старта)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Заменяем последний слой
        # Если num_classes = 1, используем BCEWithLogitsLoss (рекомендуется для бинарной классификации)
        # Если num_classes = 2, используем CrossEntropyLoss
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Тензор изображений размерности (B, C, H, W)
        Returns:
            Логиты предсказаний размерности (B, num_classes)
        """
        return self.backbone(x)

# Проверка работоспособности архитектуры при прямом запуске скрипта
if __name__ == "__main__":
    model = BaselineDetector(num_classes=1, pretrained=True)
    dummy_input = torch.randn(4, 3, 224, 224) # Batch size 4, 3 channels, 224x224
    output = model(dummy_input)
    
    print(f"Архитектура инициализирована успешно.")
    print(f"Размерность выхода модели: {output.shape}") 
    # Ожидается: torch.Size([4, 1])