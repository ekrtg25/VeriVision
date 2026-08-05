import numpy as np
import cv2
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.transforms import get_transforms

class ResNetGradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            self.target_layers = [self.model.backbone.layer4[-1]]
        else:
            self.target_layers = [target_layer]
            
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        self.transform = get_transforms(img_size=224, is_train=False)

    def generate_heatmap(self, image_path: str) -> np.ndarray:
        pil_img = Image.open(image_path).convert("RGB")
        resized_img = pil_img.resize((224, 224))
        
        rgb_img = np.float32(resized_img) / 255.0
        
        input_tensor = self.transform(pil_img).unsqueeze(0).to(next(self.model.parameters()).device)
        
        # Заставляем Grad-CAM показывать регионы наибольшего внимания слоя layer4
        # даже если итоговый вероятность низкая
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
        
        # Нормализуем маску (0..1), чтобы даже слабые отклики превратились в видимую карту
        cam_mask = grayscale_cam[0, :]
        if cam_mask.max() > cam_mask.min():
            cam_mask = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min() + 1e-8)
        
        visualization = show_cam_on_image(rgb_img, cam_mask, use_rgb=True)
        return visualization