import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm

class FlipIllusion:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda", flip_weight=0.5):
        """Initialize the modified Stable Diffusion pipeline with dual noise prediction."""
        # Load the pipeline with DDIM scheduler
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
            torch_dtype=torch.float32,
        )
        self.pipe = self.pipe.to(device)
        self.device = device
        self.flip_weight = flip_weight  # Weight for combining the noise predictions
        
    def decode_latents(self, latents):
        """Decode the latents to PIL image."""
        # Scale and decode the latents
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.pipe.vae.decode(latents).sample
        
        # Convert to PIL image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        if image.shape[0] == 1:
            image = image[0]
        return Image.fromarray(image)
    
    def encode_image(self, pil_image):
        """Encode PIL image back to latents."""
        # Convert PIL image to tensor
        np_image = np.array(pil_image)
        image = torch.from_numpy(np_image).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = 2 * image - 1  # Normalize to [-1, 1]
        image = image.to(device=self.device)
        
        # Encode image to latents
        with torch.no_grad():
            latents = self.pipe.vae.encode(image).latent_dist.sample()
        latents = 0.18215 * latents
        return latents
    
    def get_flipped_latents(self, latents):
        """Decode latents, flip the image, and re-encode."""
        # Decode the latents to image
        image = self.decode_latents(latents)
        
        # Flip the image upside down
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Re-encode the flipped image
        flipped_latents = self.encode_image(flipped_image)
        return flipped_latents
    
    def __call__(
        self,
        prompts,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        seed=None,
        flip_weight=None,
    ):
        """Generate image with combined noise prediction from original and flipped latents."""
        # Use provided flip_weight or default
        flip_weight = flip_weight if flip_weight is not None else self.flip_weight
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        prompt_embeddings = []
        uncond_embeddings = []
        negative_prompts = [negative_prompt] * len(prompts)
        
        for i, prompt in enumerate(prompts):
            # Tokenize the prompt
            text_inputs = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            
            embedding = self.pipe.text_encoder(text_input_ids)[0]
            prompt_embeddings.append(embedding)
            
            # Handle unconditional embeddings for CFG
            if guidance_scale > 1.0:
                uncond_inputs = self.pipe.tokenizer(
                    negative_prompts[i],
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_input_ids = uncond_inputs.input_ids.to(self.device)
                
                # Get unconditional embeddings
                uncond_embedding = self.pipe.text_encoder(uncond_input_ids)[0]
                uncond_embeddings.append(uncond_embedding)
        
        # Prepare initial latents
        latents_shape = (1, self.pipe.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=prompt_embeddings[0].dtype)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Prepare extra kwargs for the scheduler step
        extra_step_kwargs = self.pipe.scheduler.config.get("extra_step_kwargs", {})
        
        # Denoising loop with dual noise prediction
        for _, t in tqdm(enumerate(timesteps)):
            # Create flipped version of current latents
            flipped_latents = self.get_flipped_latents(latents)
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            flipped_model_input = torch.cat([flipped_latents] * 2) if guidance_scale > 1.0 else flipped_latents
            
            # Get noise predictions for original latents
            with torch.no_grad():
                noise_pred_original = self.pipe.unet(
                    latent_model_input, t, encoder_hidden_states=torch.cat([prompt_embeddings[0], uncond_embeddings[0]])
                ).sample
            
            # Get noise predictions for flipped latents
            with torch.no_grad():
                noise_pred_flipped = self.pipe.unet(
                    flipped_model_input, t, encoder_hidden_states=torch.cat([prompt_embeddings[1], uncond_embeddings[1]])
                ).sample
            
            # Perform guidance separately for each noise prediction
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred_original.chunk(2)
                noise_pred_original = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                flipped_pred_uncond, flipped_pred_text = noise_pred_flipped.chunk(2)
                noise_pred_flipped = flipped_pred_uncond + guidance_scale * (flipped_pred_text - flipped_pred_uncond)
            
            # Combine the noise predictions
            combined_noise_pred = (1 - flip_weight) * noise_pred_original + flip_weight * self.get_flipped_latents(noise_pred_flipped)
            
            # Compute previous sample using the scheduler
            latents = self.pipe.scheduler.step(
                model_output=combined_noise_pred,
                timestep=t,
                sample=latents,
                **extra_step_kwargs
            ).prev_sample
                
        # Decode and return the final image
        image = self.decode_latents(latents)
        return image

# Example usage
if __name__ == "__main__":
    # Initialize the modified pipeline
    dual_noise_diffusion = FlipIllusion(
        model_id="runwayml/stable-diffusion-v1-5", 
        device="cuda",
        flip_weight=0.5  # Equal weighting between original and flipped noise predictions
    )
    
    # Generate an image with the prompt
    prompt = "a photograph of an astronaut riding a horse on the moon"
    image = dual_noise_diffusion(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=30,
        seed=42,
        # Optional: override default flip_weight
        # flip_weight=0.3  # Lower weight for flipped noise
    )
    
    # Save the resulting image
    image.save("dual_noise_astronaut.png")
    print("Image generated and saved as dual_noise_astronaut.png")