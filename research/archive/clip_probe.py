
import open_clip
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()


class CLIPProbeDetector(nn.Module):
    """
    Детектор AI-контента на основе замороженного визуального энкодера CLIP
    и обучаемого линейного классификатора (Linear Probe).

    Подход основан на исследовании Ojha et al. (CVPR 2023).
    """

    def __init__(
        self,
        model_name: str = 'ViT-B-32',
        pretrained: str = 'laion2b_s34b_b79k',
        num_classes: int = 1
    ):
        super().__init__()

        # 1. Загружаем модель CLIP и трансформы для нее
        # visual_model возвращает эмбеддинг изображения
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )

        # Забираем только визуальный энкодер (текстовый нам не нужен)
        self.visual_encoder = clip_model.visual

        # 2. Замораживаем веса CLIP полностью!
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        # 3. Определяем размерность выходного вектора CLIP
        # Для ViT-B/32 это обычно 512 элементов
        embedding_dim = clip_model.visual.output_dim

        # 4. Создаем обучаемый Linear Probe (один полносвязный слой)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Препроцессированный тензор изображений (B, 3, H, W)
        """
        # Извлекаем эмбеддинги без подсчета градиентов для слоя CLIP
        with torch.no_grad():
            features = self.visual_encoder(x)

            # Если CLIP возвращает нормализованные эмбеддинги, отлично;
            # иначе нормализуем их для стабильности классификации
            features = features / features.norm(dim=-1, keepdim=True)

        # Прогоняем вектор признаков через наш единственный линейный слой
        logits = self.fc(features)
        return logits


# Быстрый Sanity Check при запуске модуля
if __name__ == "__main__":
    print("Инициализация CLIP Probe...")
    detector = CLIPProbeDetector(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k')

    # Проверяем, сколько параметров реально обучаются
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in detector.parameters())

    print(f"Всего параметров в модели: {total_params:,}")
    print(f"Обучаемых параметров (Linear Probe): {trainable_params:,}")

    dummy_input = torch.randn(2, 3, 224, 224)
    output = detector(dummy_input)
    print(f"✅ CLIPProbeDetector работает! Выходной тензор: {output.shape}")
