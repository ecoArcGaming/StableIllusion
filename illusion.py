import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any


class IllusionDiffusion(ABC):
    """
    Abstract base class for implementing various latent space manipulations
    in diffusion models to create different illusion effects.
    """
    def __init__(
        self, 
        model_id: str = "runwayml/stable-diffusion-v1-5", 
        device: str = "cuda", 
        transform_weight: float = 0.5
    ):
        """
        Initialize the diffusion pipeline with transformation capabilities.
        
        Args:
            model_id: The model ID to load from HuggingFace
            device: The device to run inference on
            transform_weight: Weight for combining original and transformed predictions
        """
        # Load the pipeline with DDIM scheduler
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
            torch_dtype=torch.float32,
        )
        self.pipe = self.pipe.to(device)
        self.device = device
        self.transform_weight = transform_weight
        
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """
        Decode latents to PIL image.
        
        Args:
            latents: The latent vectors to decode
            
        Returns:
            A PIL Image
        """
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
    
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Encode PIL image back to latents.
        
        Args:
            pil_image: The PIL image to encode
            
        Returns:
            Latent tensor representation of the image
        """
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
    
    @abstractmethod
    def transform_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Apply a transformation to the latents.
        Each subclass must implement its own transformation.
        
        Args:
            latents: The latent vectors to transform
            
        Returns:
            Transformed latent vectors
        """
        pass
    
    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        transform_weight: Optional[float] = None,
        num_images: int = 1,
        multi_transform_weights: Optional[List[float]] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate images with combined noise prediction from original and transformed latents.
        
        Args:
            prompt: The text prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            negative_prompt: Negative text prompt
            seed: Random seed for reproducibility
            transform_weight: Override default transform weight
            num_images: Number of base images to generate and blend
            multi_transform_weights: Weights for multiple transformations
            
        Returns:
            Generated PIL Image(s)
        """
        # Use provided transform_weight or default
        transform_weight = transform_weight if transform_weight is not None else self.transform_weight
        
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
        
        # Generate multiple base latents if requested
        latents_list = []
        for _ in range(num_images):
            # Generate initial latents
            latents_shape = (1, self.pipe.unet.in_channels, height // 8, width // 8)
            latents = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)
            latents = latents * self.pipe.scheduler.init_noise_sigma
            latents_list.append(latents)
        
        # Default to equal weights if not provided
        if multi_transform_weights is None and num_images > 1:
            multi_transform_weights = [1.0 / num_images] * num_images
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Prepare extra kwargs for the scheduler step
        extra_step_kwargs = self.pipe.scheduler.config.get("extra_step_kwargs", {})
        
        # Denoising loop for each set of latents
        for i, t in enumerate(timesteps):
            for idx, latents in enumerate(latents_list):
                # Create transformed version of current latents
                transformed_latents = self.transform_latents(latents)
                
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                transformed_model_input = torch.cat([transformed_latents] * 2) if guidance_scale > 1.0 else transformed_latents
                
                # Get noise predictions for original latents
                with torch.no_grad():
                    noise_pred_original = self.pipe.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample
                
                # Get noise predictions for transformed latents
                with torch.no_grad():
                    noise_pred_transformed = self.pipe.unet(
                        transformed_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample
                
                # Perform guidance separately for each noise prediction
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred_original.chunk(2)
                    noise_pred_original = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    transformed_pred_uncond, transformed_pred_text = noise_pred_transformed.chunk(2)
                    noise_pred_transformed = transformed_pred_uncond + guidance_scale * (transformed_pred_text - transformed_pred_uncond)
                
                # Transform the transformed noise prediction back to align with original
                transformed_noise_aligned = self.transform_latents(noise_pred_transformed)
                
                # Combine the noise predictions with weight
                combined_noise_pred = (1 - transform_weight) * noise_pred_original + transform_weight * transformed_noise_aligned
                
                # Compute previous sample using the scheduler
                latents_list[idx] = self.pipe.scheduler.step(
                    model_output=combined_noise_pred,
                    timestep=t,
                    sample=latents,
                    **extra_step_kwargs
                ).prev_sample
        
        # For multiple images, blend them according to weights
        if num_images > 1:
            final_latents = torch.zeros_like(latents_list[0])
            for idx, latents in enumerate(latents_list):
                final_latents += multi_transform_weights[idx] * latents
            result = self.decode_latents(final_latents)
        else:
            result = self.decode_latents(latents_list[0])
            
        return result

    def generate_multiple(
        self,
        prompt: str,
        transforms: List[Dict[str, Any]],
        blend_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate multiple images with different transformations and blend them.
        
        Args:
            prompt: The text prompt
            transforms: List of transformation parameters
            blend_weights: Weights for blending the multiple images
            **kwargs: Additional arguments for the generation process
            
        Returns:
            Blended PIL Image
        """
        if blend_weights is None:
            blend_weights = [1.0 / len(transforms)] * len(transforms)
            
        latents_list = []
        
        for transform_params in transforms:
            # Store original transform weight
            original_weight = self.transform_weight
            
            # Update transformation parameters if specified
            if 'transform_weight' in transform_params:
                self.transform_weight = transform_params['transform_weight']
            
            # Generate initial latents
            height = transform_params.get('height', kwargs.get('height', 512))
            width = transform_params.get('width', kwargs.get('width', 512))
            latents_shape = (1, self.pipe.unet.in_channels, height // 8, width // 8)
            
            # Set seed if provided
            if 'seed' in transform_params:
                torch.manual_seed(transform_params['seed'])
            
            latents = torch.randn(latents_shape, device=self.device)
            latents = latents * self.pipe.scheduler.init_noise_sigma
            
            # Generate the image
            result_latents = self._generate_latents(
                prompt=prompt,
                initial_latents=latents,
                height=height,
                width=width,
                num_inference_steps=transform_params.get('num_inference_steps', 
                                                        kwargs.get('num_inference_steps', 50)),
                guidance_scale=transform_params.get('guidance_scale', 
                                                  kwargs.get('guidance_scale', 7.5)),
                negative_prompt=transform_params.get('negative_prompt', 
                                                   kwargs.get('negative_prompt', None))
            )
            
            latents_list.append(result_latents)
            
            # Restore original transform weight
            self.transform_weight = original_weight
        
        # Blend the latents according to weights
        final_latents = torch.zeros_like(latents_list[0])
        for idx, latents in enumerate(latents_list):
            final_latents += blend_weights[idx] * latents
            
        return self.decode_latents(final_latents)
    
    def _generate_latents(
        self,
        prompt: str,
        initial_latents: torch.Tensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Helper method to generate latents for a single image.
        
        Args:
            prompt: The text prompt
            initial_latents: Starting point for diffusion
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            negative_prompt: Negative text prompt
            
        Returns:
            Generated latents
        """
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
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Prepare extra kwargs for the scheduler step
        extra_step_kwargs = self.pipe.scheduler.config.get("extra_step_kwargs", {})
        
        # Start with the initial latents
        latents = initial_latents
        
        # Denoising loop with transformation
        for i, t in enumerate(timesteps):
            # Create transformed version of current latents
            transformed_latents = self.transform_latents(latents)
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            transformed_model_input = torch.cat([transformed_latents] * 2) if guidance_scale > 1.0 else transformed_latents
            
            # Get noise predictions for original latents
            with torch.no_grad():
                noise_pred_original = self.pipe.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample
            
            # Get noise predictions for transformed latents
            with torch.no_grad():
                noise_pred_transformed = self.pipe.unet(
                    transformed_model_input, t, encoder_hidden_states=text_embeddings
                ).sample
            
            # Perform guidance separately for each noise prediction
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred_original.chunk(2)
                noise_pred_original = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                transformed_pred_uncond, transformed_pred_text = noise_pred_transformed.chunk(2)
                noise_pred_transformed = transformed_pred_uncond + guidance_scale * (transformed_pred_text - transformed_pred_uncond)
            
            # Transform the transformed noise prediction back to align with original
            transformed_noise_aligned = self.transform_latents(noise_pred_transformed)
            
            # Combine the noise predictions with weight
            combined_noise_pred = (1 - self.transform_weight) * noise_pred_original + self.transform_weight * transformed_noise_aligned
            
            # Compute previous sample using the scheduler
            latents = self.pipe.scheduler.step(
                model_output=combined_noise_pred,
                timestep=t,
                sample=latents,
                **extra_step_kwargs
            ).prev_sample
                
        return latents