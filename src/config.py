# File: src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/stable-diffusion-2-base")
    IMG_SIZE = int(os.getenv("IMG_SIZE", 512))
    LATENT_SIZE = int(os.getenv("LATENT_SIZE", 64))
    
    # Training
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-5))
    MIXED_PRECISION = os.getenv("MIXED_PRECISION", "fp16")
    
    # Paths
    DATASET_PATH = os.getenv("DATASET_PATH", "/data/train")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/models/finetuned")
    LOG_DIR = os.getenv("LOG_DIR", "/logs")
    
    # Hardware
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    DISTRIBUTED = os.getenv("DISTRIBUTED", "false").lower() == "true"
    
    # Inference
    NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", 50))
    GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", 7.5))
    
    @staticmethod
    def print_config():
        import logging
        logger = logging.getLogger(__name__)
        for key, value in vars(Config).items():
            if not key.startswith("__") and not callable(value):
                logger.info(f"{key}: {value}")