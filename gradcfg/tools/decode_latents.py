import argparse

import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.utils as vutils

SCALE_FACTOR = 0.18215  # Match the latent scaling used during training.


def decode_latents_to_long_image(
    latent_path, model_name, out_path, orientation="vertical"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the VAE.
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    vae.eval()

    # 2. Load the latent tensor.
    print(f"Loading latents from: {latent_path}")
    latents = torch.load(latent_path, map_location=device)

    # Add a batch dimension when a single latent image was saved.
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)

    print(f"Latents shape: {latents.shape}")  # [B, 4, 64, 64] for SD 512x512

    with torch.no_grad():
        # Training multiplies encoded latents by the latent scaling factor.
        # Undo this scaling before decoding.
        latents = latents / SCALE_FACTOR

        # 3. Decode VAE outputs in [-1, 1].
        images = vae.decode(latents).sample

    # 4. Map decoded pixels to [0, 1].
    images = (images.clamp(-1, 1) + 1) / 2.0  # [B, 3, H, W]

    # 5. Arrange images into a strip.
    if orientation == "vertical":
        # Arrange images in one column.
        grid = vutils.make_grid(images, nrow=1, padding=0)
    else:
        # Arrange images in one row.
        grid = vutils.make_grid(images, nrow=images.shape[0])

    # 6. Convert to uint8 and save.
    grid = grid.mul(255).clamp(0, 255).byte().cpu()
    ndarr = grid.permute(1, 2, 0).numpy()
    img = Image.fromarray(ndarr)
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved long image to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode saved SD latents to a long image"
    )
    parser.add_argument(
        "--model_name",
        default="sd_model/tiny-sd/",
        help="Stable Diffusion base model path (same as training)",
    )
    parser.add_argument(
        "--latent_path",
        default="ckpts/model_ckpt/dream_booth_tiny/mix/training_noises/img.pt",
        help="Path to saved latent tensor (img.pt)",
    )
    parser.add_argument(
        "--out_path",
        default="mix_results/mix_ori.png",
        help="Where to save the decoded long image",
    )
    parser.add_argument(
        "--orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Strip orientation: vertical or horizontal",
    )

    args = parser.parse_args()
    decode_latents_to_long_image(
        latent_path=args.latent_path,
        model_name=args.model_name,
        out_path=args.out_path,
        orientation=args.orientation,
    )
