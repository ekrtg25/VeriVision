import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import warnings

# Отключаем лишние логи от Hugging Face
warnings.filterwarnings("ignore")

class DIREAnalyzer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[sys] Загрузка DIRE (Stable Diffusion) на {self.device}...")
        
        # Загружаем пайплайн. Для маков используем float16 (если mps) или float32 для cpu
        dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            safety_checker=None
        ).to(self.device)
        
        # Настраиваем DDIM планировщик для детерминированной реконструкции
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Трансформации для перевода картинки в тензор
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def get_reconstruction_error(self, image_path, num_inference_steps=20):
        """
        Прогоняет картинку через VAE и диффузию, возвращает ошибку реконструкции (MAE).
        Чем ниже ошибка, тем выше вероятность, что это диффузионный дипфейк.
        """
        # 1. Подготовка оригинального изображения
        raw_img = Image.open(image_path).convert("RGB")
        original_tensor = self.transform(raw_img).unsqueeze(0).to(self.device, dtype=self.pipe.dtype)
        
        with torch.no_grad():
            # 2. Кодируем в латентное пространство (VAE)
            # Умножаем на scale_factor, как этого требует архитектура SD
            latents = self.pipe.vae.encode(original_tensor * 2.0 - 1.0).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
            
            # 3. Упрощенная имитация DDIM Inversion -> Denoising
            # Добавляем шум (отправляем в будущее)
            noise = torch.randn_like(latents)
            timesteps = torch.tensor([self.pipe.scheduler.config.num_train_timesteps - 1], device=self.device)
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Декодируем обратно (пытаемся восстановить)
            reconstructed_latents = self.pipe(
                prompt="", 
                latents=noisy_latents, 
                num_inference_steps=num_inference_steps,
                output_type="latent"
            ).images
            
            # Переводим восстановленные латенты обратно в пиксели
            reconstructed_latents = 1 / self.pipe.vae.config.scaling_factor * reconstructed_latents
            reconstructed_tensor = self.pipe.vae.decode(reconstructed_latents).sample
            
            # Нормализуем обратно в диапазон [0, 1]
            reconstructed_tensor = (reconstructed_tensor / 2 + 0.5).clamp(0, 1)
            
        # 4. Вычисляем Mean Absolute Error (MAE)
        # Если картинка диффузионная, разница между original и reconstructed будет минимальной
        error = torch.nn.functional.l1_loss(original_tensor, reconstructed_tensor).item()
        
        return error