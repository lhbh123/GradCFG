import argparse

import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.utils as vutils

SCALE_FACTOR = 0.18215  # 和训练时保持一致


def decode_latents_to_long_image(latent_path, model_name, out_path, orientation="vertical"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载 VAE
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    vae.eval()

    # 2. 加载 latent 张量
    print(f"Loading latents from: {latent_path}")
    latents = torch.load(latent_path, map_location=device)

    # 可能保存的是 [4, H, W] 的单张图，这里统一成 [B, 4, H, W]
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)

    print(f"Latents shape: {latents.shape}")  # [B, 4, 64, 64] for SD 512x512

    with torch.no_grad():
        # 注意：训练时 encode 用的是  latents = vae.encode(image).latent_dist.sample() * 0.18215
        # 解码时需要除回来
        latents = latents / SCALE_FACTOR

        # 3. VAE 解码 -> [-1, 1] 区间
        images = vae.decode(latents).sample

    # 4. 映射到 [0, 1]
    images = (images.clamp(-1, 1) + 1) / 2.0  # [B, 3, H, W]

    # 5. 拼成长图
    if orientation == "vertical":
        # 一列，多行（竖着的长图）
        grid = vutils.make_grid(images, nrow=1, padding=0)
    else:
        # 一行，多列（横着的长图）
        grid = vutils.make_grid(images, nrow=images.shape[0])

    # 6. 转成 uint8 并保存
    grid = grid.mul(255).clamp(0, 255).byte().cpu()
    ndarr = grid.permute(1, 2, 0).numpy()
    img = Image.fromarray(ndarr)
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved long image to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode saved SD latents to a long image")
    parser.add_argument(
        "--model_name",
        default="sd_model/tiny-sd/",
        help="Stable Diffusion base model path (same as training)"
    )
    parser.add_argument(
        "--latent_path",
        default="ckpts/model_ckpt/dream_booth_tiny/mix/training_noises/img.pt",
        help="Path to saved latent tensor (img.pt)"
    )
    parser.add_argument(
        "--out_path",
        default="mix_results/mix_ori.png",
        help="Where to save the decoded long image"
    )
    parser.add_argument(
        "--orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="拼接方向：vertical = 竖长图，horizontal = 横长图"
    )

    args = parser.parse_args()
    decode_latents_to_long_image(
        latent_path=args.latent_path,
        model_name=args.model_name,
        out_path=args.out_path,
        orientation=args.orientation,
    )
