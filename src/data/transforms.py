"""
Robust Forensics Augmentations for Train and Validation.
Forces model to evaluate compressed/blurred images consistently.
"""

import io
import random
from PIL import Image, ImageFilter
from torchvision import transforms

DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD = [0.229, 0.224, 0.225]


class RandomJPEGCompression:
    def __init__(self, quality_min: int = 45, quality_max: int = 95, p: float = 0.6):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(self.quality_min, self.quality_max)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img


class RandomGaussianBlur:
    def __init__(self, p: float = 0.3, radius_max: float = 1.5):
        self.p = p
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(0.3, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


def get_robust_train_transforms(image_size: int = 518) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomJPEGCompression(quality_min=40, quality_max=90, p=0.7),
        RandomGaussianBlur(p=0.4, radius_max=1.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])


def get_val_transforms(image_size: int = 518) -> transforms.Compose:
    # Валидация тоже идет со сжатием и шумом для честной проверки OOD
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        RandomJPEGCompression(quality_min=50, quality_max=90, p=0.5),
        RandomGaussianBlur(p=0.25, radius_max=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])