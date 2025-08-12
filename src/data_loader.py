# File: src/data_loader.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPTokenizer
import logging

logger = logging.getLogger(__name__)

class TextImageDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, size=512):
        self.tokenizer = tokenizer
        self.size = size
        self.image_paths = []
        self.captions = []
        
        # Load dataset - expected structure: dataset_path/{image_files} and captions.txt
        captions_file = os.path.join(dataset_path, "captions.txt")
        if not os.path.exists(captions_file):
            raise FileNotFoundError(f"Captions file not found at {captions_file}")
        
        with open(captions_file, "r") as f:
            for line in f:
                img_file, caption = line.strip().split(",", 1)
                img_path = os.path.join(dataset_path, img_file.strip())
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.captions.append(caption.strip())
        
        if len(self.image_paths) == 0:
            raise RuntimeError("No valid images found in dataset directory")
        
        logger.info(f"Loaded {len(self)} samples from {dataset_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            # Center crop and resize
            width, height = image.size
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            image = image.crop((left, top, right, bottom))
            image = image.resize((self.size, self.size))
            
            # Convert to tensor and normalize
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
            
            # Tokenize caption
            tokenized = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": image,
                "input_ids": tokenized.input_ids[0]
            }
        except Exception as e:
            logger.error(f"Error loading {img_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))  # Skip bad sample