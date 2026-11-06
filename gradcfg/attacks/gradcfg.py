"""Full-batch GradCFG with lower overhead for throughput-oriented runs."""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from gradcfg.attacks.base import (
    DEFAULT_NUM_TIMESTEPS,
    DiffusionReconstructor,
    parse_args,
)
from gradcfg.common.artifacts import prepare_run_directory
from gradcfg.common.optimization import tv_loss


def parse_gradcfg_args(argv=None):
    return parse_args(argv, tv_lambda_default=0.5, tv_every_default=5)


class GradCFGReconstructor(DiffusionReconstructor):
    def __init__(self, args):
        super().__init__(args)
        self.vae.requires_grad_(False)
        # The model weights remain fixed. Use a detached leaf sharing their
        # storage for the observed parameter-gradient calculation.
        self.attack_parameter = self.model_init.flat_param.detach().requires_grad_(True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def _init_text_embeddings(self):
        embeddings = super()._init_text_embeddings()
        del self.text_encoder
        del self.tokenizer
        return embeddings

    def _model_inputs(self):
        timestep = self.timesteps * DEFAULT_NUM_TIMESTEPS
        return timestep, self.add_noise(
            self.dummy_images, self.gaussian_images, timestep
        )

    def _constrain_timesteps(self):
        with torch.no_grad():
            t_min = max(0.001, min(self.args.t_min, 1.0))
            t_max = max(0.001, min(self.args.t_max, 1.0))
            if t_max <= t_min:
                t_max = min(1.0, t_min + 1.0 / DEFAULT_NUM_TIMESTEPS)
            self.timesteps.clamp_(t_min, t_max)

    def reconstruction_loss(self, epoch):
        timestep, noisy_image = self._model_inputs()
        text = self.text_emb.expand(self.args.batch_size, -1, -1)
        prediction = self.model_init(
            noisy_image, timestep, text, flat_param=self.attack_parameter
        ).sample
        training_loss = F.mse_loss(prediction, self.gaussian_images)
        gradient = torch.autograd.grad(
            training_loss, self.attack_parameter, create_graph=True
        )[0]
        main_loss = 1 - F.cosine_similarity(gradient, self.model_grad, dim=0, eps=1e-20)

        diversity = self._compute_similarity_loss()
        total_loss = main_loss
        if diversity is not None and epoch <= self.args.switch_epoch:
            total_loss = total_loss + self.args.ind_lambda * diversity

        tv = torch.zeros(
            (), device=self.dummy_images.device, dtype=self.dummy_images.dtype
        )
        if (
            self.args.tv_lambda > 0
            and self.args.tv_every > 0
            and epoch % self.args.tv_every == 0
        ):
            decoded = self.vae.decode(self.dummy_images / 0.18215).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            tv = tv_loss(decoded)
            total_loss = total_loss + self.args.tv_lambda * tv
        return total_loss, main_loss, diversity, tv

    def train(self):
        save_dir = (
            Path("ckpts/g_inv_dream_booth")
            / self.args.model_name
            / self.args.label_name
            / f"{self.args.start_epoch}_{self.args.batch_size}_{self.args.reini_time}"
        )
        save_dir = prepare_run_directory(self.args, save_dir)
        attack_variables = (
            self.dummy_images,
            self.text_emb,
            self.gaussian_images,
            self.timesteps,
        )
        self.logger.info("Fast full-batch attack: batch=%d", self.args.batch_size)
        run_start = time.perf_counter()
        log_start = run_start
        last_logged_epoch = -1
        log_every = self.args.log_every
        if log_every < 1 or self.args.save_every < 0:
            raise ValueError(
                "--log_every must be positive and --save_every nonnegative"
            )
        if self.dummy_images.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.dummy_images.device)

        for epoch in range(self.args.num_epochs):
            if (
                self.args.reini_time > 0
                and epoch > 0
                and epoch % self.args.reini_time == 0
            ):
                self.reinit_gaussian_images()

            self.optimizer.zero_grad(set_to_none=True)
            total_loss, main_loss, diversity, tv = self.reconstruction_loss(epoch)
            torch.autograd.backward(total_loss, inputs=attack_variables)
            self.optimizer.step()

            self._constrain_timesteps()

            if epoch % log_every == 0 or epoch == self.args.num_epochs - 1:
                if self.dummy_images.is_cuda:
                    torch.cuda.synchronize(self.dummy_images.device)
                now = time.perf_counter()
                steps = epoch - last_logged_epoch
                peak_allocated = (
                    torch.cuda.max_memory_allocated(self.dummy_images.device) / 2**30
                    if self.dummy_images.is_cuda
                    else 0.0
                )
                peak_reserved = (
                    torch.cuda.max_memory_reserved(self.dummy_images.device) / 2**30
                    if self.dummy_images.is_cuda
                    else 0.0
                )
                self.logger.info(
                    "Epoch %04d | Loss: %.4f | Main Loss: %.4f | Diversity Loss: %.4f | "
                    "TV: %.6f | avg_step_s: %.3f | elapsed_s: %.1f | "
                    "peak_step_allocated_gib: %.2f | peak_step_reserved_gib: %.2f",
                    epoch,
                    total_loss.item(),
                    main_loss.item(),
                    diversity.item() if diversity is not None else 0.0,
                    tv.item(),
                    (now - log_start) / steps,
                    now - run_start,
                    peak_allocated,
                    peak_reserved,
                )
                log_start = time.perf_counter()
                last_logged_epoch = epoch
                if self.dummy_images.is_cuda:
                    torch.cuda.reset_peak_memory_stats(self.dummy_images.device)

            if epoch == self.args.num_epochs - 1 or (
                self.args.save_every > 0 and epoch % self.args.save_every == 0
            ):
                torch.save(
                    self.dummy_images.detach().cpu(),
                    save_dir / f"dummy_images_epoch{epoch}.pth",
                )
                torch.save(
                    self.text_emb.detach().cpu(),
                    save_dir / f"text_emb_epoch{epoch}.pth",
                )
                image_dir = save_dir / f"epoch_{epoch}_images"
                image_dir.mkdir(exist_ok=True)
                with torch.no_grad():
                    decoded = self.vae.decode(
                        self.dummy_images.detach() / 0.18215
                    ).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    save_image(
                        decoded,
                        image_dir / "recons.png",
                        nrow=1,
                        normalize=False,
                        padding=0,
                    )
                log_start = time.perf_counter()
                if self.dummy_images.is_cuda:
                    torch.cuda.reset_peak_memory_stats(self.dummy_images.device)


if __name__ == "__main__":
    GradCFGReconstructor(parse_gradcfg_args()).train()
