import sys
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from transformers import Dinov2Model


class DINOv2ForensicStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        hidden_dim = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self.patch_anomaly = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token).squeeze(-1)
        return logits


def main():
    model_path = "models/perceptual_student.pth"
    test_img_path = sys.argv[1] if len(sys.argv) > 1 else "data/robust_v1/fake/fake_00925.jpg"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Загрузка модели на {device} из {model_path}...")

    model = DINOv2ForensicStudent().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"[+] Анализ изображения: {test_img_path}")
    img = Image.open(test_img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    verdict = "AI_GENERATED" if prob >= 0.5 else "REAL_PHOTO"
    print("\n" + "=" * 45)
    print(f" РЕЗУЛЬТАТ ДЕТЕКЦИИ DINOv2")
    print("=" * 45)
    print(f" Вероятность генерации (AI): {prob:.2%}")
    print(f" Вердикт модели:            {verdict}")
    print("=" * 45)


if __name__ == "__main__":
    main()
