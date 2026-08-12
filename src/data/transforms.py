from torchvision import transforms

def get_transforms(img_size: int = 224, is_train: bool = True):
    """
    Возвращает пайплайн трансформаций для изображений.
    
    Args:
        img_size: Размер, к которому приводится изображение.
        is_train: Флаг режима обучения (включает аугментации).
    """
    # Нормализация по стандартам ImageNet (так как используем ConvNeXt-Tiny и визуальный бэкбон)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])