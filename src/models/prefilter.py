"""
Zero-Shot Semantic Content Prefilter for VeriVision MoE v3.0
Distinguishes stylized CGI/Art/Screenshots from camera captures.
"""

from typing import Dict
import torch
import open_clip
from PIL import Image


class ContentPrefilter:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.device = torch.device("cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        self.prompts = {
            "real_photo": "a natural camera photo, realistic lighting flaws, authentic textures",
            "ai_generated": "an AI generated realistic image, photorealistic diffusion output",
            "digital_art": "3D CGI render, animation, cartoon, digital art, stylized illustration",
            "screenshot": "a software UI screenshot, computer screen, digital text layout",
        }
        self.labels = list(self.prompts.keys())

        # Precompute text embeddings for zero-latency inference
        text_tokens = self.tokenizer(list(self.prompts.values()))
        with torch.inference_mode():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def classify_semantics(self, image_pil: Image.Image) -> Dict[str, float]:
        """
        Returns softmax similarity distribution across semantic categories.
        """
        img_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            image_features = self.model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            scores = logits[0].tolist()

        return dict(zip(self.labels, scores))