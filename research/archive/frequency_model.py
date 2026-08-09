import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FFTTransform(nn.Module):
    """
    Преобразует RGB-изображение в частотный спектр мощности с помощью 2D FFT.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Тензор изображений (B, C, H, W)
        Returns:
            Тензор спектра мощности той же размерности (B, C, H, W)
        """
        # 1. Переводим в Grayscale для чистоты спектра (или работаем по каждому каналу)
        # Для простоты вычисляем FFT по каждому RGB каналу отдельно.
        
        # 2. Применяем 2D Быстрое Преобразование Фурье
        # dim=(-2, -1) означает, что преобразуем пространственные оси (H, W)
        fft_complex = torch.fft.fft2(x, dim=(-2, -1))
        
        # 3. Сдвигаем нулевую (низкую) частоту в центр картинки (fftshift)
        # Низкие частоты (общие формы) окажутся в центре, высокие (детали/артефакты) — по краям.
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        
        # 4. Вычисляем амплитудный спектр (Magnitude = |FFT|)
        magnitude = torch.abs(fft_shifted)
        
        # 5. Логарифмическое масштабирование: log(1 + Magnitude)
        # Разброс значений у FFT огромный (от 0 до 10^6). Логарифм сглаживает диапазон для нейросети.
        spectrum = torch.log1p(magnitude)
        
        # 6. Нормализация каждого спектра в диапазон [0, 1] для стабильности градиентов
        min_val = spectrum.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = spectrum.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        spectrum = (spectrum - min_val) / (max_val - min_val + 1e-8)
        
        return spectrum


class FrequencyDetector(nn.Module):
    """
    Детектор AI-контента на основе частотного анализа.
    Сначала извлекает спектр Фурье, затем классифицирует его легкой сетью (ResNet18).
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        
        # Модуль преобразования Фурье (не имеет обучаемых параметров)
        self.fft_layer = FFTTransform()
        
        # Для анализа спектра достаточно более легкой сети ResNet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Заменяем классификатор
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Шаг 1: Конвертируем входные картинки в карты частот
        freq_representation = self.fft_layer(x)
        
        # Шаг 2: Прогоняем спектр через классификатор
        logits = self.backbone(freq_representation)
        
        return logits


# Проверка работы слоя при прямом запуске
if __name__ == "__main__":
    dummy_img = torch.randn(2, 3, 224, 224) # Batch size 2
    model = FrequencyDetector(num_classes=1, pretrained=True)
    
    output = model(dummy_img)
    print("✅ FrequencyDetector успешно инициализирован.")
    print(f"Размерность входа: {dummy_img.shape}")
    print(f"Размерность выхода: {output.shape}") # Ожидается: torch.Size([2, 1])