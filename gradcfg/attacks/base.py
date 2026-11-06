from gradcfg.common.artifacts import gradient_path
from gradcfg.common.optimization import set_seed

import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from gradcfg.common.reparam import ReparamModule


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_IMAGE_SIZE = 64
DEFAULT_NUM_TIMESTEPS = 1000
DEFAULT_CHANNELS = 4


class DiffusionReconstructor:
    """Diffusion Model-based Image Reconstructor"""

    def __init__(self, args: argparse.Namespace):
        from diffusers import AutoencoderKL
        from transformers import CLIPTokenizer, CLIPTextModel

        self.args = args
        self.logger = self._setup_logger()

        # Set seed before initializing any components
        set_seed(args.seed)

        # Initialize components
        self.vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae").to(
            DEVICE
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.model_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.model_path, subfolder="text_encoder"
        ).to(DEVICE)
        self.model_init, self.model_grad = self._build_models()
        self.text_emb = self._init_text_embeddings()
        self.dummy_images = self._init_dummy_data()
        self.gaussian_images = self._init_gaussian_data()
        self.timesteps = self._init_timesteps()
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self.dummy_images], "lr": args.lr_img, "betas": (0.8, 0.9)},
                {"params": [self.text_emb], "lr": args.lr_txt, "betas": (0.8, 0.9)},
                {
                    "params": [self.gaussian_images],
                    "lr": args.lr_noi,
                    "betas": (0.8, 0.9),
                },
                {"params": [self.timesteps], "lr": args.lr_t, "betas": (0.8, 0.9)},
            ],
        )

        self.logger.info(f"Random seed set to: {args.seed}")
        self.logger.info(
            f"t range: [{self.args.t_min:.4f}, {self.args.t_max:.4f}] (normalized)"
        )
        self.logger.info(f"TV weight: {self.args.tv_lambda:.2e}")

    def _setup_logger(self) -> logging.Logger:
        """Configure logging system"""
        logger = logging.getLogger("ReconstructionLogger")
        logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers when initialized repeatedly.
        if logger.handlers:
            return logger

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File handler
        log_dir = Path("logs/recons")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"{self.args.model_name}_seed{self.args.seed}.log"
        )
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def _build_models(self) -> tuple[ReparamModule, Any]:
        """Initialize model triplets for reconstruction"""
        print(f"Loading model from {self.args.model_path}")
        # Model initialization
        from diffusers import UNet2DConditionModel
        from diffusers.models.attention_processor import AttnProcessor

        model_init = UNet2DConditionModel.from_pretrained(
            self.args.model_path, subfolder="unet"
        )
        # Resolve the checkpoint directory by run name and label.
        ckpt_dir = (
            Path("./ckpts/model_ckpt")
            / self.args.model_name
            / f"{self.args.label_name}"
        )
        model_init.load_state_dict(
            torch.load(ckpt_dir / f"epoch_{self.args.start_epoch}.pt")
        )
        observed_path = getattr(self.args, "observed_gradient_path", None)
        model_grad = torch.load(
            observed_path or gradient_path(ckpt_dir, self.args.start_epoch),
            map_location=DEVICE,
        )
        print(torch.norm(model_grad).item())

        for model in [model_init]:
            model.set_attn_processor(AttnProcessor())

        # Device placement
        model_init.to(DEVICE).eval()

        model_init = ReparamModule(model_init)

        return model_init, model_grad

    def alpha_bar_cont(self, t: torch.Tensor, beta_start=0.00085, beta_end=0.012):
        """Continuous alpha_bar(t) approximation"""
        k_t = np.sqrt(beta_start) + t / DEFAULT_NUM_TIMESTEPS * (
            np.sqrt(beta_end) - np.sqrt(beta_start)
        )
        k_0 = torch.zeros_like(k_t) + np.sqrt(beta_start)
        alpha_bar = torch.exp(
            DEFAULT_NUM_TIMESTEPS
            * (-(k_t**3) + k_0**3)
            / (3 * (np.sqrt(beta_end) - np.sqrt(beta_start)))
        )
        return alpha_bar

    def _init_text_embeddings(self) -> torch.Tensor:
        with torch.no_grad():
            text_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        embeddings.requires_grad_(True)
        return embeddings

    def _init_dummy_data(self) -> torch.Tensor:
        """Initialize trainable dummy images (latent space)"""
        tensor = (
            torch.randn(self.args.batch_size, 3, 512, 512).to(DEVICE)
            * self.args.init_scale
        )
        with torch.no_grad():
            tensor = self.vae.encode(tensor).latent_dist.sample() * 0.18215

        tensor.requires_grad_(True)
        return tensor

    def _init_gaussian_data(self) -> torch.Tensor:
        """Initialize trainable gaussian images (noise)"""
        tensor = torch.randn(
            self.args.batch_size,
            DEFAULT_CHANNELS,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_IMAGE_SIZE,
        ).to(DEVICE)
        tensor.requires_grad_(True)
        return tensor

    def _init_timesteps(self) -> torch.Tensor:
        """
        Initialize continuous timesteps normalized to [0, 1].
        The configured bounds determine the initial integer index range.
        """
        # Clamp the initialization range to valid bounds.
        t_min = max(0.001, min(self.args.t_min, 1.0))
        t_max = max(0.001, min(self.args.t_max, 1.0))
        if t_max <= t_min:
            t_max = min(1.0, t_min + 1.0 / DEFAULT_NUM_TIMESTEPS)

        low = int(t_min * DEFAULT_NUM_TIMESTEPS)
        high = int(t_max * DEFAULT_NUM_TIMESTEPS)

        tensor = (
            torch.randint(
                low,
                max(high, low + 1),
                (self.args.batch_size,),
                device=DEVICE,
            ).float()
            / DEFAULT_NUM_TIMESTEPS
        )

        tensor.requires_grad_(True)
        return tensor

    def reinit_gaussian_images(self):
        """Re-initialize gaussian images (noise)"""
        with torch.no_grad():
            new_random = torch.randn_like(self.gaussian_images)
            new_data = new_random

            self.gaussian_images.data.copy_(new_data)
            if self.gaussian_images.grad is not None:
                self.gaussian_images.grad.detach_()
                self.gaussian_images.grad.zero_()

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        sqrt_alpha_prod = self.alpha_bar_cont(timesteps) ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alpha_bar_cont(timesteps)) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def _compute_similarity_loss(self) -> Tensor | None:
        """Compute the pairwise similarity penalty within the batch."""
        batch_size = self.dummy_images.size(0)
        if batch_size < 2:
            return

        clamped_img = torch.sigmoid(self.dummy_images)
        flattened = clamped_img.view(batch_size, -1)

        flattened = F.normalize(flattened, p=2, dim=-1)
        sim_matrix = torch.mm(flattened, flattened.T)

        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1)
        pair_similarities = sim_matrix[triu_indices[0], triu_indices[1]]

        similarity_loss = torch.mean(torch.abs(pair_similarities))
        return similarity_loss


def parse_args(
    argv=None, *, tv_lambda_default=0.5, tv_every_default=5
) -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Diffusion Model Reconstruction")

    parser.add_argument(
        "--model_name",
        type=str,
        default="dream_booth_tiny",
        help="Name of the diffusion model",
    )
    parser.add_argument(
        "--label_name",
        type=str,
        default="dog",
        help="Name of the label",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="sd_model/tiny-sd",
        help="Path of the diffusion model",
    )
    parser.add_argument(
        "--ind_lambda",
        type=float,
        default=1e-2,
        help="Weight for diversity regularization",
    )
    parser.add_argument(
        "--lr_img",
        type=float,
        default=1e-1,
        help="Learning rate for dummy images",
    )
    parser.add_argument(
        "--lr_txt",
        type=float,
        default=1e-3,
        help="Learning rate for text embeddings",
    )
    parser.add_argument(
        "--lr_noi",
        type=float,
        default=1e-1,
        help="Learning rate for noise",
    )
    parser.add_argument(
        "--lr_t",
        type=float,
        default=1e-2,
        help="Learning rate for timesteps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Size of batches",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="epoch of start (for loading client ckpt/grad)",
    )
    parser.add_argument(
        "--switch_epoch",
        type=int,
        default=100,
        help="epoch to switch off diversity loss",
    )
    parser.add_argument(
        "--reini_time",
        type=int,
        default=100,
        help="reinitialize noise every N epochs; <=0 disables reinitialization",
    )
    parser.add_argument(
        "--init_scale",
        type=float,
        default=5e-1,
        help="Initialization scale for dummy images",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4001,
        help="Total training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Normalized timestep bounds used during optimization.
    parser.add_argument(
        "--t_min",
        type=float,
        default=0.001,
        help="Lower bound (normalized, in [0,1]) of timesteps during reconstruction.",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=0.999,
        help="Upper bound (normalized, in [0,1]) of timesteps during reconstruction.",
    )
    # Total variation regularization.
    parser.add_argument(
        "--tv_lambda",
        type=float,
        default=tv_lambda_default,
        help="Weight for TV loss (structure prior on decoded images).",
    )
    parser.add_argument(
        "--tv_every",
        type=int,
        default=tv_every_default,
        help="Compute TV loss every N epochs (set 1 to compute every epoch).",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional exact run output directory; use separate paths for seeds/ablations",
    )
    parser.add_argument(
        "--observed_gradient_path",
        dest="observed_gradient_path",
        default=None,
        help="Optional observed parameter signal; otherwise load the captured gradient",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save every N epochs; 0 saves only the final epoch",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log timing and peak VRAM every N epochs",
    )
    args = parser.parse_args(argv)
    if args.log_every < 1 or args.save_every < 0:
        parser.error("--log_every must be positive and --save_every nonnegative")
    if not 0.001 <= args.t_min < args.t_max <= 0.999:
        parser.error("Timestep bounds must satisfy 0.001 <= t_min < t_max <= 0.999")
    return args
