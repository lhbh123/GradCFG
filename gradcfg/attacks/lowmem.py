"""GradCFG with exact two-pass batch gradient matching and chunked TV."""

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
from gradcfg.common.lowmem import backward_aggregate_cosine
from gradcfg.common.optimization import tv_loss


class LowMemoryDiffusionReconstructor(DiffusionReconstructor):
    def __init__(self, args):
        super().__init__(args)
        self.vae.requires_grad_(False)
        # The model weights are fixed during inversion. This leaf shares their
        # storage, avoiding the full parameter copy made by torch.cat each step.
        self.attack_parameter = self.model_init.flat_param.detach().requires_grad_(True)

    def _init_text_embeddings(self):
        embeddings = super()._init_text_embeddings()
        del self.text_encoder
        del self.tokenizer
        return embeddings

    def _sample_loss(self, index, detached):
        image = self.dummy_images[index:index + 1]
        noise = self.gaussian_images[index:index + 1]
        timestep = self.timesteps[index:index + 1]
        text = self.text_emb
        if detached:
            image, noise, timestep, text = (value.detach() for value in (image, noise, timestep, text))
        timestep = timestep * DEFAULT_NUM_TIMESTEPS
        noisy_image = self.add_noise(image, noise, timestep)
        prediction = self.model_init(
            noisy_image, timestep, text, flat_param=self.attack_parameter
        ).sample
        return F.mse_loss(prediction, noise)

    def _backward_regularizers(self, epoch):
        tv_value = torch.zeros((), device=self.dummy_images.device, dtype=self.dummy_images.dtype)
        if self.args.tv_lambda > 0 and self.args.tv_every > 0 and epoch % self.args.tv_every == 0:
            for index in range(self.args.batch_size):
                decoded = self.vae.decode(self.dummy_images[index:index + 1] / 0.18215).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                sample_tv = tv_loss(decoded)
                torch.autograd.backward(
                    self.args.tv_lambda * sample_tv / self.args.batch_size,
                    inputs=(self.dummy_images,),
                )
                tv_value += sample_tv.detach() / self.args.batch_size

        diversity = self._compute_similarity_loss()
        if diversity is not None and epoch <= self.args.switch_epoch:
            torch.autograd.backward(
                self.args.ind_lambda * diversity, inputs=(self.dummy_images,)
            )
        return diversity.detach() if diversity is not None else None, tv_value

    def _backward_step(self, epoch):
        diversity, tv = self._backward_regularizers(epoch)
        attack_variables = (
            self.dummy_images, self.text_emb, self.gaussian_images, self.timesteps
        )
        main_loss = backward_aggregate_cosine(
            self.attack_parameter, self.model_grad, self.args.batch_size,
            self._sample_loss, attack_variables,
        )
        total_loss = main_loss
        if diversity is not None and epoch <= self.args.switch_epoch:
            total_loss = total_loss + self.args.ind_lambda * diversity
        if self.args.tv_lambda > 0 and self.args.tv_every > 0 and epoch % self.args.tv_every == 0:
            total_loss = total_loss + self.args.tv_lambda * tv
        return total_loss, main_loss, diversity, tv

    def train(self):
        save_dir = (
            Path("ckpts/g_inv_dream_booth") / self.args.model_name / self.args.label_name
            / f"{self.args.start_epoch}_{self.args.batch_size}_{self.args.reini_time}"
        )
        save_dir = prepare_run_directory(self.args, save_dir)
        self.logger.info("Low-memory two-pass attack: batch=%d, microbatch=1", self.args.batch_size)
        run_start = time.perf_counter()
        ema_epoch_time = None

        for epoch in range(self.args.num_epochs):
            epoch_start = time.perf_counter()
            if self.dummy_images.is_cuda:
                torch.cuda.reset_peak_memory_stats(self.dummy_images.device)
            if self.args.reini_time > 0 and epoch > 0 and epoch % self.args.reini_time == 0:
                self.reinit_gaussian_images()

            self.optimizer.zero_grad(set_to_none=True)
            total_loss, main_loss, diversity, tv = self._backward_step(epoch)
            self.optimizer.step()

            with torch.no_grad():
                t_min = max(0.001, min(self.args.t_min, 1.0))
                t_max = max(0.001, min(self.args.t_max, 1.0))
                if t_max <= t_min:
                    t_max = min(1.0, t_min + 1.0 / DEFAULT_NUM_TIMESTEPS)
                self.timesteps.clamp_(t_min, t_max)

            epoch_time = time.perf_counter() - epoch_start
            ema_epoch_time = epoch_time if ema_epoch_time is None else 0.9 * ema_epoch_time + 0.1 * epoch_time
            eta = int(ema_epoch_time * (self.args.num_epochs - epoch - 1))
            elapsed = int(time.perf_counter() - run_start)
            peak_allocated = (
                torch.cuda.max_memory_allocated(self.dummy_images.device) / 2**30
                if self.dummy_images.is_cuda else 0.0
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved(self.dummy_images.device) / 2**30
                if self.dummy_images.is_cuda else 0.0
            )
            if epoch % self.args.log_every == 0 or epoch == self.args.num_epochs - 1:
                self.logger.info(
                    "Epoch %04d | Loss: %.4f | Main Loss: %.4f | Diversity Loss: %.4f | "
                    "TV: %.6f | epoch_time_s: %.3f | eta_s: %d | elapsed_s: %d | "
                    "peak_step_allocated_gib: %.2f | peak_step_reserved_gib: %.2f",
                    epoch, total_loss.item(), main_loss.item(),
                    diversity.item() if diversity is not None else 0.0,
                    tv.item(), epoch_time, eta, elapsed, peak_allocated, peak_reserved,
                )

            if epoch == self.args.num_epochs - 1 or (self.args.save_every > 0 and epoch % self.args.save_every == 0):
                torch.save(self.dummy_images.detach().cpu(), save_dir / f"dummy_images_epoch{epoch}.pth")
                torch.save(self.text_emb.detach().cpu(), save_dir / f"text_emb_epoch{epoch}.pth")
                image_dir = save_dir / f"epoch_{epoch}_images"
                image_dir.mkdir(exist_ok=True)
                with torch.no_grad():
                    decoded_images = []
                    for index in range(self.args.batch_size):
                        decoded = self.vae.decode(
                            self.dummy_images[index:index + 1].detach() / 0.18215
                        ).sample
                        decoded_images.append((decoded / 2 + 0.5).clamp(0, 1).cpu())
                    save_image(torch.cat(decoded_images), image_dir / "recons.png",
                               nrow=1, normalize=False, padding=0)


if __name__ == "__main__":
    LowMemoryDiffusionReconstructor(parse_args()).train()
