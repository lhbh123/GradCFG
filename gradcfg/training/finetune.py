from gradcfg.training.statistics import calculate_weight_changes
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from pathlib import Path
import json

TIME = 1000


class SimpleDataset(Dataset):
    def __init__(self, data_dir, prompt, tokenizer, size=512, num_samples=None):
        all_files = sorted(
            [f for f in Path(f"{data_dir}/{prompt}").glob("*") if f.is_file()],
            key=lambda x: x.name
        )
        if num_samples is not None:
            all_files = all_files[:num_samples]
        self.image_paths = all_files
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel = self.transform(image)
        tokens = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        return {"pixel_values": pixel, "input_ids": tokens.input_ids.squeeze()}


def train(args):
    if args.seed is not None:
        from gradcfg.common.optimization import set_seed
        set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=TIME,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear"
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    dataset = SimpleDataset(args.data_dir, args.prompt, tokenizer, num_samples=args.num_samples)
    if len(dataset) == 0:
        raise ValueError("No training images found; check --data_dir and --prompt.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)

    check_epochs = set(args.check_epochs)

    output_dir = Path(f"{args.output_dir}/{args.prompt}")
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_save_path = os.path.join(output_dir, "training_noises")
    os.makedirs(noise_save_path, exist_ok=True)

    all_noises = []
    all_timesteps_list = []

    ori_path = output_dir / "epoch_0.pt"
    torch.save(unet.state_dict(), ori_path)

    for epoch in range(args.epochs):
        unet.eval()
        epoch_loss = 0.0

        if epoch in check_epochs:
            torch.save(unet.state_dict(), output_dir / f"epoch_{epoch}.pt")

        for batch_idx, batch in enumerate(dataloader):
            pixels = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixels).latent_dist.sample() * 0.18215
                text_emb = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)

            # Use the full scheduler range for every training and capture step.
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            )

            all_noises.append(noise.cpu().detach().clone())
            all_timesteps_list.append(timesteps.cpu().detach().clone())

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(noisy_latents, timesteps, text_emb).sample

            loss = F.mse_loss(noise_pred, noise)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            if epoch in check_epochs and batch_idx == 0:
                gradients = []
                for param in unet.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.detach().clone().cpu())
                    else:
                        gradients.append(None)

                flat_grads = []
                for grad in gradients:
                    if grad is not None:
                        flat_grads.append(grad.view(-1))

                if flat_grads:
                    all_gradients = torch.cat(flat_grads)
                else:
                    all_gradients = torch.tensor([])

                torch.save(
                    all_gradients,
                    os.path.join(output_dir, f"gradients_epoch_{epoch}.pt")
                )

            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch}/{args.epochs} | Average Loss: {avg_loss:.4f}")

    final_path = output_dir / f"epoch_{args.epochs}.pt"
    torch.save(unet.state_dict(), final_path)
    print(f"Saved final UNet weights to {final_path}")

    torch.save(all_noises, os.path.join(noise_save_path, "all_noises.pt"))
    torch.save(all_timesteps_list, os.path.join(noise_save_path, "all_timesteps.pt"))
    torch.save(latents.cpu().detach().clone(), os.path.join(noise_save_path, "img.pt"))
    torch.save(text_emb.cpu().detach().clone(), os.path.join(noise_save_path, "text_emb.pt"))

    initial_weights = torch.load(ori_path)
    final_weights = torch.load(final_path)

    weight_stats = calculate_weight_changes(initial_weights, final_weights)

    print(f"\nWeight Change Analysis:")
    print(f"Norm of Initial Weights: {weight_stats['norm_initial']:.4f}")
    print(f"Norm of Weight Changes: {weight_stats['norm_delta']:.4f}")
    print(f"Ratio (Δ/Initial): {weight_stats['delta_ratio']:.6f}")

    stats_path = output_dir / "weight_changes.json"
    with open(stats_path, 'w') as f:
        json.dump(weight_stats, f, indent=4)
    print(f"Saved weight change statistics to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="sd_model/tiny-sd")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--prompt", default="backpack")
    parser.add_argument("--output_dir", default="ckpts/model_ckpt/dream_booth_tiny")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed; omitted preserves historical behavior")

    # Select epochs at which to save the model and first-batch gradient.
    parser.add_argument(
        "--check_epochs",
        type=int,
        nargs="+",
        default=[0],
        help="Epochs for checkpoint and gradient capture, e.g. --check_epochs 50 100 150",
    )

    args = parser.parse_args()
    train(args)
