"""Run Inv-Rev using two model states, or ordinary CFG with one model."""
import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("disparity", "cfg"), default="disparity", help="Two model disparity (Inv-Rev) or single model CFG")
    parser.add_argument("--model", default="sd_model/tiny-sd")
    parser.add_argument("--prompt")
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--text", required=True, type=Path, help="Recovered text embedding tensor")
    parser.add_argument("--original-checkpoint", required=True, type=Path)
    parser.add_argument("--finetuned-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-inv", type=float, default=1)
    parser.add_argument("--guidance-sam", type=float, default=2)
    parser.add_argument("--guidance-sam-values", nargs="+", type=float,
                        help="Sample multiple disparity scales from one DDIM inversion")
    parser.add_argument("--output-dir", type=Path, help="Directory for guidance_<scale>.png in sweep mode")
    parser.add_argument("--device", help="Default: CUDA when available, otherwise CPU")
    args = parser.parse_args(argv)
    args.prompt = args.prompt or ""
    args.output = args.output or Path("outputs/inv_rev.png")
    if args.mode == "disparity" and args.finetuned_checkpoint is None:
        parser.error("--finetuned-checkpoint is required for Inv-Rev")
    if args.num_steps < 2 or args.num_steps > 1000:
        parser.error("--num-steps must be between 2 and 1000")
    if args.guidance_sam_values is not None:
        if args.mode != "disparity" or args.output_dir is None:
            parser.error("--guidance-sam-values requires --mode disparity and --output-dir")
        if len(set(args.guidance_sam_values)) != len(args.guidance_sam_values):
            parser.error("Guidance scales must be unique")
    return args


def expand_embeddings(embeddings, batch_size):
    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)
    if embeddings.ndim != 3 or embeddings.shape[0] not in (1, batch_size):
        raise ValueError("Text embeddings must have shape [1 or batch, sequence, dimension].")
    return embeddings.expand(batch_size, -1, -1)


def main():
    args = parse_args()
    needed = [args.latents, args.text, args.original_checkpoint]
    if args.mode == "disparity":
        needed.append(args.finetuned_checkpoint)
    for path in needed:
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment artifact is missing: {path}. See README.md for training and checkpoint setup.")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from gradcfg.sampling.inv_rev import ddim_inversion, ddim_sampling, inversion_based_sampling
    from gradcfg.tools.images import compose

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    original = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(device)
    original.load_state_dict(torch.load(args.original_checkpoint, map_location=device))
    original.eval()
    finetuned = None
    if args.mode == "disparity":
        finetuned = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(device)
        finetuned.load_state_dict(torch.load(args.finetuned_checkpoint, map_location=device))
        finetuned.eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device).eval()
    tokens = tokenizer([args.prompt, ""], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        prompt_embedding, empty_embedding = text_encoder(tokens.input_ids.to(device))[0].chunk(2)
        latents = torch.load(args.latents, map_location=device)
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        if latents.ndim != 4 or latents.shape[0] < 1:
            raise ValueError("Latents must have shape [batch, channels, height, width].")
        recovered = expand_embeddings(torch.load(args.text, map_location=device), latents.shape[0])
        print(f"Mean text embedding cosine: {F.cosine_similarity(prompt_embedding.expand_as(recovered).flatten(1), recovered.flatten(1), dim=1).mean().item():.4f}")
        unconditional = empty_embedding.expand(latents.shape[0], -1, -1)
        if args.guidance_sam_values is None:
            generated = inversion_based_sampling(
                target_latents=latents, text_embeddings=recovered,
                uncond_embeddings=unconditional, original_unet=original,
                finetuned_unet=finetuned, scheduler=scheduler,
                num_steps=args.num_steps, guidance_scale_inv=args.guidance_inv,
                guidance_scale_sam=args.guidance_sam, device=device,
            )
            outputs = [(args.output, generated)]
        else:
            inverted = ddim_inversion(
                target_latents=latents, unet=original, text_embeddings=recovered,
                uncond_embeddings=unconditional, scheduler=scheduler,
                guidance_scale=args.guidance_inv,
                num_inversion_steps=args.num_steps, device=device,
            )
            outputs = []
            for scale in args.guidance_sam_values:
                generated = ddim_sampling(
                    initial_latents=inverted, unet_ori=original, unet_fin=finetuned,
                    text_embeddings=recovered, scheduler=scheduler,
                    guidance_scale=scale, num_inference_steps=args.num_steps,
                    device=device, uncond_embeddings=unconditional,
                )
                outputs.append((args.output_dir / f"guidance_{scale:g}.png", generated))
        for path, generated in outputs:
            images = (vae.decode(generated / 0.18215).sample / 2 + 0.5).clamp(0, 1)
            arrays = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            path.parent.mkdir(parents=True, exist_ok=True)
            compose([Image.fromarray(array) for array in arrays]).save(path)
            print(f"Saved {path} (mode={args.mode})")


if __name__ == "__main__":
    main()
