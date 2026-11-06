from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from diffusers import DDIMScheduler


@torch.no_grad()
def ddim_sampling(
    initial_latents: torch.Tensor,
    unet_ori: torch.nn.Module,
    unet_fin: torch.nn.Module | None,
    text_embeddings: torch.Tensor,
    scheduler: DDIMScheduler,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    device: str = "cuda",
    uncond_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sample with model disparity, or with ordinary CFG if unet_fin is None.

    Inv-Rev uses the same recovered text for the initial and final model.
    Single-model CFG uses conditional and unconditional text on unet_ori.

    Args:
        initial_latents: Initial noise tensor (batch_size, 4, H, W)
        unet: UNet model for noise prediction
        text_embeddings: Conditional text embeddings (batch_size * 2, seq_len, emb_dim)
        uncond_embeddings: Unconditional embeddings (batch_size * 2, seq_len, emb_dim)
        scheduler: DDIM scheduler for timestep management
        guidance_scale: Scale for classifier-free guidance
        num_inference_steps: Number of denoising steps
        device: Device for computation

    Returns:
        Generated image latents (batch_size, 4, H, W)
    """
    batch_size = initial_latents.shape[0]
    if unet_fin is None and uncond_embeddings is None:
        raise ValueError("Single-model CFG requires unconditional embeddings.")

    # Configure scheduler timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = initial_latents.to(device).detach().clone()

    # Denoising loop
    for t in tqdm(scheduler.timesteps, desc="DDIM Sampling"):
        # Expand latents for both conditional and unconditional outputs
        latent_model_input = scheduler.scale_model_input(latents, t)

        # Predict noise residual
        if unet_fin is not None and guidance_scale == 0:
            noise_pred = unet_ori(
                latent_model_input, t.expand(batch_size), text_embeddings
            ).sample
        elif unet_fin is not None and guidance_scale == 1:
            noise_pred = unet_fin(
                latent_model_input, t.expand(batch_size), text_embeddings
            ).sample
        else:
            noise_pred_cond = (unet_fin if unet_fin is not None else unet_ori)(
                latent_model_input, t.expand(batch_size), text_embeddings
            ).sample
            noise_pred_uncond = unet_ori(
                latent_model_input,
                t.expand(batch_size),
                text_embeddings if unet_fin is not None else uncond_embeddings,
            ).sample
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        # Update latents
        latents = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

    return latents


@torch.no_grad()
def ddim_inversion(
    target_latents: torch.Tensor,
    unet: torch.nn.Module,
    text_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    scheduler: DDIMScheduler,
    guidance_scale: float = 7.5,
    num_inversion_steps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Perform DDIM inversion to reconstruct noise latents from images.

    Args:
        target_latents: Target image latents (batch_size, 4, H, W)
        unet: UNet model for noise prediction
        text_embeddings: Conditional text embeddings (batch_size * 2, seq_len, emb_dim)
        uncond_embeddings: Unconditional embeddings (batch_size * 2, seq_len, emb_dim)
        scheduler: DDIM scheduler for timestep management
        guidance_scale: Scale for classifier-free guidance
        num_inversion_steps: Number of inversion steps
        device: Device for computation

    Returns:
        Reconstructed noise latents (batch_size, 4, H, W)
    """
    batch_size = target_latents.shape[0]

    # Combine conditional and unconditional embeddings
    conditioning = torch.cat([uncond_embeddings, text_embeddings])

    # Configure scheduler timesteps in reverse order
    scheduler.set_timesteps(num_inversion_steps, device=device)
    reverse_timesteps = reversed(scheduler.timesteps)
    latents = target_latents.to(device).detach().clone()

    # Inversion loop
    for i, t in enumerate(tqdm(reverse_timesteps, desc="DDIM Inversion")):
        # Skip last step for reconstruction
        if i >= num_inversion_steps - 1:
            continue

        # Expand latents for both conditional and unconditional outputs
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict noise residual
        noise_pred = unet(
            latent_model_input, t.expand(batch_size * 2), conditioning
        ).sample

        # Apply classifier-free guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # Approximate inversion using current and next timestep
        current_t = max(0, t.item() - (1000 // num_inversion_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = scheduler.alphas_cumprod[current_t]
        alpha_t_next = scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
            alpha_t_next.sqrt() / alpha_t.sqrt()
        ) + (1 - alpha_t_next).sqrt() * noise_pred

    return latents


@torch.no_grad()
def inversion_based_sampling(
    target_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    original_unet: torch.nn.Module,
    finetuned_unet: torch.nn.Module | None,
    scheduler: DDIMScheduler,
    num_steps: int = 50,
    guidance_scale_inv: float = 7.5,
    guidance_scale_sam: float = 7.5,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Invert with the original model, then sample with Inv-Rev or single-model CFG.

    Args:
        target_latents: Target image latents (batch_size, 4, H, W)
        text_embeddings: Conditional text embeddings
        uncond_embeddings: Unconditional embeddings
        original_unet: Pretrained UNet for inversion
        finetuned_unet: Finetuned UNet for sampling
        scheduler: DDIM scheduler for timestep management
        num_steps: Number of inversion/sampling steps
        guidance_scale: Scale for classifier-free guidance
        device: Device for computation

    Returns:
        Generated image latents (batch_size, 4, H, W)
    """
    # Step 1: Inversion with original model
    reconstructed_latents = ddim_inversion(
        target_latents=target_latents,
        unet=original_unet,
        text_embeddings=text_embeddings,
        uncond_embeddings=uncond_embeddings,
        scheduler=scheduler,
        guidance_scale=guidance_scale_inv,
        num_inversion_steps=num_steps,
        device=device,
    )

    # Step 2: Sampling with finetuned model
    generated_latents = ddim_sampling(
        initial_latents=reconstructed_latents,
        unet_ori=original_unet,
        unet_fin=finetuned_unet,
        text_embeddings=text_embeddings,
        scheduler=scheduler,
        guidance_scale=guidance_scale_sam,
        num_inference_steps=num_steps,
        device=device,
        uncond_embeddings=uncond_embeddings,
    )

    return generated_latents
