import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from PIL import Image
import numpy as np

class FlippedStableDiffusion:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        """Initialize the modified Stable Diffusion pipeline."""
        # Load the pipeline with DDIM scheduler
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.pipe = self.pipe.to(device)
        self.device = device
        
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
    
    def flip_latents(self, latents):
        """Decode latents, flip the image, and re-encode."""
        # Decode the latents to image
        image = self.decode_latents(latents)
        
        # Flip the image upside down
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Re-encode the flipped image
        flipped_latents = self.encode_image(flipped_image)
        return flipped_latents
        
    def custom_step(self, sample, timestep, clean_latents, extra_step_kwargs, eta):
        """Custom step function that flips the image at each denoising step."""
        # Original DDIM step
        step_output = self.pipe.scheduler.step(
            model_output=clean_latents,
            timestep=timestep,
            sample=sample,
            eta=eta,
            **extra_step_kwargs
        )
        
        # Get the denoised latents from the step output
        prev_sample = step_output.prev_sample
        
        # Flip the latents by decoding, flipping, and re-encoding
        flipped_sample = self.flip_latents(prev_sample)
        
        # Return modified step output
        return step_output._replace(prev_sample=flipped_sample)
    
    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        seed=None,
        eta=0.0,
    ):
        """Generate image with upside-down flipping at each timestep."""
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        # Prepare text embeddings
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Get text embeddings
        text_embeddings = self.pipe.text_encoder(text_input_ids)[0]
        
        # Get unconditional embeddings for classifier-free guidance
        if guidance_scale > 1.0:
            max_length = text_input_ids.shape[-1]
            uncond_input = self.pipe.tokenizer(
                [""] if negative_prompt is None else [negative_prompt],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.pipe.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare initial latents
        latents_shape = (1, self.pipe.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Prepare extra kwargs for the scheduler step
        extra_step_kwargs = self.pipe.scheduler.config.get("extra_step_kwargs", {})
        
        # Denoising loop with custom step function
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Predict noise residual
            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample
            
            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous latent
            step_output = self.custom_step(
                sample=latents,
                timestep=t,
                clean_latents=noise_pred, 
                extra_step_kwargs=extra_step_kwargs,
                eta=eta
            )
            
            # Update latents
            latents = step_output.prev_sample
                
        # Decode and return the final image
        image = self.decode_latents(latents)
        return image

# Example usage
if __name__ == "__main__":
    # Initialize the modified pipeline
    flipped_diffusion = FlippedStableDiffusion(model_id="runwayml/stable-diffusion-v1-5", device="cuda")
    
    # Generate an image with the prompt
    prompt = "a photograph of an astronaut riding a horse on the moon"
    image = flipped_diffusion(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=30,
        seed=42
    )
    
    # Save the resulting image
    image.save("flipped_astronaut.png")
    print("Image generated and saved as flipped_astronaut.png")