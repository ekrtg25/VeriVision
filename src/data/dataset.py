import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset

class ForensicsDataset(Dataset):
    """
    Dataset для загрузки изображений из структуры:
    root_dir/
      ├── real/  (label 0)
      └── fake/  (label 1)
    """
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        self._load_samples()

    def _load_samples(self):
        classes = {'real': 0, 'fake': 1}
        
        for class_name, label in classes.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Папка не найдена: {class_dir}")
                continue
                
            # Проходим по всем файлам внутри папки real / fake
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((file_path, label))

        if not self.samples:
            print(f"Внимание: Не найдено изображений в {self.root_dir.resolve()}")
        else:
            print(f"Загружено {len(self.samples)} изображений из {self.root_dir.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label