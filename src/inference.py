# File: src/inference.py
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from config import Config
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self, model_path=None):
        self.device = "cuda" if Config.USE_GPU and torch.cuda.is_available() else "cpu"
        self.model_path = model_path or Config.MODEL_NAME
        
        # Load model
        if model_path:
            logger.info(f"Loading fine-tuned model from {model_path}")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                Config.MODEL_NAME,
                unet=torch.load(os.path.join(model_path, "unet/pytorch_model.bin")),
                safety_checker=None,
                torch_dtype=torch.float16 if Config.MIXED_PRECISION else torch.float32
            ).to(self.device)
        else:
            logger.info("Loading base model")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                Config.MODEL_NAME,
                torch_dtype=torch.float16 if Config.MIXED_PRECISION else torch.float32,
                safety_checker=None
            ).to(self.device)
        
        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

    def generate_image(self, prompt):
        try:
            with torch.autocast(self.device) if Config.MIXED_PRECISION else dummy_context():
                image = self.pipeline(
                    prompt,
                    num_inference_steps=Config.NUM_INFERENCE_STEPS,
                    guidance_scale=Config.GUIDANCE_SCALE,
                    width=Config.IMG_SIZE,
                    height=Config.IMG_SIZE
                ).images[0]
            return image
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError("Image generation failed") from e

class dummy_context:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        pass

# API Example (for integration with Flask/FastAPI)
# generator = ImageGenerator("/models/finetuned")
# image = generator.generate_image("a futuristic cityscape at sunset")
# image.save("output.png")