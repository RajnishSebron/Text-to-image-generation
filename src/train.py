# File: src/train.py
import torch
import logging
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from config import Config
from data_loader import TextImageDataset
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)

def setup_distributed():
    if Config.DISTRIBUTED:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0

def main():
    # Initialize distributed training if enabled
    rank = setup_distributed()
    
    # Load components
    tokenizer = CLIPTokenizer.from_pretrained(Config.MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(Config.MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(Config.MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(Config.MODEL_NAME, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(Config.MODEL_NAME, subfolder="scheduler")
    
    # Move to GPU if available
    device = torch.device(f"cuda:{rank}" if Config.USE_GPU and torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)
    
    # Freeze non-UNet components
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Create dataset and loader
    dataset = TextImageDataset(Config.DATASET_PATH, tokenizer, size=Config.IMG_SIZE)
    loader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    # Prepare for distributed training
    if Config.DISTRIBUTED:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, 
            device_ids=[rank],
            output_device=rank
        )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=Config.LEARNING_RATE
    )
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=Config.NUM_EPOCHS * len(loader)
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if Config.MIXED_PRECISION else None
    
    # Tensorboard logging
    writer = SummaryWriter(Config.LOG_DIR) if rank == 0 else None
    
    # Training loop
    global_step = 0
    for epoch in range(Config.NUM_EPOCHS):
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}", disable=rank != 0)
        for batch in progress_bar:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=device
            ).long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
            
            # Predict noise residual
            with torch.cuda.amp.autocast(enabled=Config.MIXED_PRECISION):
                noise_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states
                ).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            lr_scheduler.step()
            global_step += 1
            
            # Logging
            if rank == 0 and global_step % 10 == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], global_step)
                progress_bar.set_postfix(loss=loss.item())
    
    # Save final model
    if rank == 0:
        unet.module.save_pretrained(os.path.join(Config.OUTPUT_DIR, "unet"))
        logger.info(f"Training complete. Model saved to {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Config.print_config()
    main()