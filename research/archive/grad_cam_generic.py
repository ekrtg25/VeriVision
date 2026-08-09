import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple

class GradCAM:
    """
    Класс для генерации тепловых карт Grad-CAM для сверточных нейросетей.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Обученная PyTorch модель.
            target_layer: Сверточный слой, с которого снимаются градиенты (обычно последний conv-слой).
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Регистрируем хуки (hooks) для перехвата активаций и градиентов
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Навешиваем хуки на целевой слой
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Генерирует нормализованную тепловую карту (2D numpy array [0, 1]).
        
        Args:
            input_tensor: Тензор изображения (1, C, H, W)
        """
        self.model.eval()
        
        # 1. Прямой проход (Forward pass)
        output = self.model(input_tensor)
        
        # 2. Обнуляем градиенты
        self.model.zero_grad()
        
        # 3. Обратный проход (Backward pass) относительно предсказанного логита
        output.backward(retain_graph=True)
        
        # 4. Вычисляем глобальное среднее градиентов (веса α_k)
        # self.gradients имеет размерность (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 5. Линейная комбинация активаций с весами
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # 6. Применяем ReLU (оставляем только позитивные вклады в классификацию)
        cam = torch.relu(cam)
        
        # Переводим в numpy
        cam = cam.squeeze().cpu().numpy()
        
        # 7. Нормализуем в диапазон [0, 1]
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam


def overlay_heatmap(
    heatmap: np.ndarray, 
    original_img: np.ndarray, 
    alpha: float = 0.5, 
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Накладывает тепловую карту на оригинальное RGB-изображение.
    
    Args:
        heatmap: 2D массив (H_feat, W_feat) со значениями от 0 до 1.
        original_img: Исходная картинка NumPy (H, W, 3) в формате RGB с значениями [0, 255].
        alpha: Прозрачность наложения тепловой карты (0.5 = 50%).
        colormap: Палитра цвета OpenCV (по умолчанию JET: синий -> зеленый -> красный).
        
    Returns:
        Кортеж: (цветная_теплокарта, наложенное_изображение) в формате RGB.
    """
    # Изменяем размер теплокарты под размер оригинального изображения
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Переводим нормализованную карту в формат 0-255 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Применяем цветовое окрашивание (ColorMap)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # OpenCV работает в BGR, переводим в RGB
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    # Смешиваем оригинальную картинку и тепловую карту
    overlay = cv2.addWeighted(original_img, 1 - alpha, color_heatmap, alpha, 0)
    
    return color_heatmap, overlay


if __name__ == "__main__":
    from src.models.baseline_cnn import BaselineDetector
    
    print("🧪 Проверка модуля Grad-CAM...")
    model = BaselineDetector(num_classes=1, pretrained=True)
    
    # Для ResNet50 целевой слой — последний блок resnet: model.backbone.layer4
    target_layer = model.backbone.layer4
    
    cam_extractor = GradCAM(model, target_layer)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = cam_extractor.generate_heatmap(dummy_input)
    
    dummy_orig_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, result_overlay = overlay_heatmap(heatmap, dummy_orig_img)
    
    print(f"✅ Grad-CAM успешно сгенерирован!")
    print(f"Размерность heatmap: {heatmap.shape}")
    print(f"Размерность итогового overlay: {result_overlay.shape}")