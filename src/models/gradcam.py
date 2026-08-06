# src/models/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

class ResNetGradCAM:  # Сохраняем имя класса для совместимости
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Регистрируем хуки на последнюю стадию ConvNeXt (features[7])
        target_layer = self.model.model.features[7]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, image_path):
        raw_image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(raw_image).unsqueeze(0).to(next(self.model.parameters()).device)

        output = self.model(input_tensor)
        self.model.zero_grad()

        # Градиент по выходу
        output.backward(torch.ones_like(output))

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, raw_image.size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        raw_np = np.array(raw_image)
        superimposed = np.uint8(heatmap * 0.4 + raw_np * 0.6)
        return Image.fromarray(superimposed)