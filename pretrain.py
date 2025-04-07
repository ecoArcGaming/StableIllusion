import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.transforms.functional import rotate
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import random
import wandb  # Optional for logging

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the transformer model
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),  # Group norm for stability
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class LatentFlipTransformer(nn.Module):
    def __init__(self, latent_channels=4, base_channels=64, t_emb_dim=256):
        super().__init__()
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        
        # Main processing blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
        ])
        
        # Time conditioning injectors
        self.time_blocks = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, base_channels)
            ) for _ in range(4)
        ])
        
        # Output projection
        self.conv_out = nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t_emb):
        # Process time embedding
        t_emb = self.time_embed(t_emb)
        
        # Initial features
        h = self.conv_in(x)
        
        # Process through blocks with time conditioning
        for block, time_block in zip(self.blocks, self.time_blocks):
            # Apply residual block
            h = block(h)
            
            # Add time conditioning
            t_out = time_block(t_emb)
            h = h + t_out.unsqueeze(-1).unsqueeze(-1)
        
        # Output projection
        out = self.conv_out(h)
        
        # Skip connection from input to output (helps preserve information)
        return out + x

# Custom wrapper for ImageNet dataset that provides flipped pairs
class FlippedImageNetWrapper(Dataset):
    def __init__(self, imagenet_dataset):
        """
        Args:
            imagenet_dataset: torchvision ImageNet dataset instance
        """
        self.dataset = imagenet_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # We don't need the class label
        # Create 180-degree flipped version
        flipped_image = rotate(image, 180)
        
        return image, flipped_image

# Training utilities
def encode_to_latent(vae, image_tensor):
    """Convert image tensor to latent space"""
    # Normalize to [-1, 1]
    if image_tensor.min() >= 0 and image_tensor.max() <= 1:
        image_tensor = 2.0 * image_tensor - 1.0
    
    latent = vae.encode(image_tensor).latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    
    return latent

def decode_from_latent(vae, latent_tensor):
    """Convert latent tensor to image space"""
    # Scale latents
    latent_tensor = latent_tensor / vae.config.scaling_factor
    
    # Decode to image
    image = vae.decode(latent_tensor).sample
    
    # Normalize to [0, 1]
    image = (image + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)
    
    return image

def get_timestep_embedding(timesteps, embedding_dim=256):
    """
    Create sinusoidal timestep embeddings like in original diffusion papers
    """
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # Zero pad if uneven
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    
    return emb

def main():
    # Configuration
    config = {
        "imagenet_root": "/path/to/imagenet",  # Path to ImageNet dataset
        "output_dir": "./output/latent_flip_model",
        "model_name": "runwayml/stable-diffusion-v1-5",
        "batch_size": 8,
        "learning_rate": 1e-4,
        "epochs": 30,
        "image_size": 512,
        "num_samples": 50000,  # Number of ImageNet samples to use
        "use_wandb": False,  # Set to True to log with Weights & Biases
    }
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Initialize wandb (optional)
    if config["use_wandb"]:
        wandb.init(project="latent-flip-transformer", config=config)
    
    # Load SD pipeline and extract VAE and noise scheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        config["model_name"], 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.to(device)
    
    vae = pipe.vae
    scheduler = pipe.scheduler
    
    # Set VAE to eval mode as we don't want to train it
    vae.eval()
    
    # Data preparation - using ImageNet directly
    transform = transforms.Compose([
        transforms.Resize(config["image_size"]),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
    ])
    
    # Load ImageNet dataset
    try:
        imagenet_dataset = ImageNet(
            root=config["imagenet_root"],
            split='train',
            transform=transform
        )
        
        # If we want to use a subset of ImageNet
        if config["num_samples"] < len(imagenet_dataset):
            indices = random.sample(range(len(imagenet_dataset)), config["num_samples"])
            imagenet_subset = Subset(imagenet_dataset, indices)
        else:
            imagenet_subset = imagenet_dataset
            
        # Wrap with our flipping dataset
        dataset = FlippedImageNetWrapper(imagenet_subset)
        
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        print("Falling back to a demo dataset for testing...")
        
        # Fallback: create a small synthetic dataset for testing
        class DemoDataset(Dataset):
            def __init__(self, size=1000, image_size=512):
                self.size = size
                self.image_size = image_size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Create a random image with some structure
                img = torch.rand(3, self.image_size, self.image_size)
                # Add some shapes to make it more interesting
                img[:, 100:400, 100:400] = torch.rand(3, 1, 1) * 0.5
                # Flip for the target
                flipped = rotate(img, 180)
                return img, flipped
                
        dataset = DemoDataset(size=config["num_samples"], image_size=config["image_size"])
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Initialize model
    model = LatentFlipTransformer().to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    
    # Loss functions
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, (original_images, flipped_images) in enumerate(progress_bar):
            original_images = original_images.to(device)
            flipped_images = flipped_images.to(device)
            
            # Get latent representations
            with torch.no_grad():
                original_latents = encode_to_latent(vae, original_images)
                target_latents = encode_to_latent(vae, flipped_images)
            
            # Sample timesteps uniformly for cross-timestep training
            timesteps = torch.randint(
                0, len(scheduler.timesteps), (original_latents.shape[0],), device=device
            )
            
            # Get noise scales for each timestep
            sigmas = scheduler.sigmas[timesteps]
            
            # Prepare time embeddings
            t_emb = get_timestep_embedding(timesteps).to(device)
            
            # Add noise according to the timestep
            noise = torch.randn_like(original_latents)
            noisy_latents = original_latents + sigmas.view(-1, 1, 1, 1) * noise
            
            # Forward pass through model
            predicted_latents = model(noisy_latents, t_emb)
            
            # Calculate loss components
            # L1 loss between predicted and target latents
            main_loss = l1_loss(predicted_latents, target_latents)
            
            # Add pixel-space consistency (decode and compare occasionally)
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    pred_images = decode_from_latent(vae, predicted_latents[:4])  # Use only a few samples
                    target_images = decode_from_latent(vae, target_latents[:4])
                
                # Visual sanity check and logging
                if batch_idx % 200 == 0:
                    comparison = torch.cat([
                        original_images[:4],
                        target_images,
                        pred_images
                    ], dim=0)
                    grid = make_grid(comparison, nrow=4, normalize=True)
                    save_image(grid, f"{config['output_dir']}/epoch{epoch}_batch{batch_idx}.png")
                    
                    if config["use_wandb"]:
                        wandb.log({
                            "examples": wandb.Image(grid),
                            "step": epoch * len(dataloader) + batch_idx
                        })
            
            # Optimize
            optimizer.zero_grad()
            main_loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Log
            epoch_loss += main_loss.item()
            progress_bar.set_postfix(loss=main_loss.item())
            
            if config["use_wandb"]:
                wandb.log({
                    "batch_loss": main_loss.item(),
                    "step": epoch * len(dataloader) + batch_idx
                })
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_epoch_loss:.6f}")
        
        if config["use_wandb"]:
            wandb.log({
                "epoch": epoch, 
                "epoch_loss": avg_epoch_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == config["epochs"] - 1:
            checkpoint_path = f"{config['output_dir']}/checkpoint_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
    # Final model save
    torch.save(model.state_dict(), f"{config['output_dir']}/latent_flip_final.pt")
    print(f"Training complete. Model saved to {config['output_dir']}/latent_flip_final.pt")
    
    # Close wandb
    if config["use_wandb"]:
        wandb.finish()
    
    return model

# Evaluation function
def evaluate_model(model, vae, test_dataset, device, batch_size=16):
    # Create a DataLoader for evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    model.eval()
    total_latent_loss = 0
    
    with torch.no_grad():
        for original_images, flipped_images in tqdm(test_loader, desc="Evaluating"):
            original_images = original_images.to(device)
            flipped_images = flipped_images.to(device)
            
            # Get latent representations
            original_latents = encode_to_latent(vae, original_images)
            target_latents = encode_to_latent(vae, flipped_images)
            
            # Zero timestep for evaluation (clean latents)
            t_emb = get_timestep_embedding(torch.zeros(original_latents.shape[0], device=device, dtype=torch.long))
            
            # Forward pass
            predicted_latents = model(original_latents, t_emb)
            
            # Calculate loss
            loss = F.mse_loss(predicted_latents, target_latents)
            total_latent_loss += loss.item()
    
    avg_latent_loss = total_latent_loss / len(test_loader)
    print(f"Evaluation - Average Latent MSE Loss: {avg_latent_loss:.6f}")
    return avg_latent_loss

# Function to use the trained model with Stable Diffusion
def generate_with_flip(model, pipeline, prompt, flip_at_step=25, num_inference_steps=50):
    # Set up callback
    @torch.no_grad()
    def callback_fn(i, t, latents):
        # Apply flip at the specified step
        if i == flip_at_step:
            # Get timestep embedding
            timestep = torch.tensor([t], device=latents.device)
            t_emb = get_timestep_embedding(timestep)
            
            # Apply transformation
            return model(latents, t_emb)
        return latents
    
    # Run pipeline with callback
    result = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        callback=callback_fn,
        callback_steps=1
    )
    
    return result

# Example usage
if __name__ == "__main__":
    # Train the model
    trained_model = main()
    
    # Or load a previously trained model
    # model = LatentFlipTransformer().to(device)
    # model.load_state_dict(torch.load("./output/latent_flip_model/latent_flip_final.pt"))
    
    # Load SD pipeline for generation
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    # Generate an image with the flip transformation
    result = generate_with_flip(
        model=trained_model,
        pipeline=pipe,
        prompt="a symmetrical castle with reflection in water",
        flip_at_step=25  # Apply flip halfway through the generation
    )
    
    # Save the result
    result.images[0].save("flipped_generation.png")