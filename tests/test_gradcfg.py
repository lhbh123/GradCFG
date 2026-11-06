"""Numerical equivalence of the throughput attack and original objective."""

import unittest
from types import SimpleNamespace

import torch

from gradcfg.common.reparam import ReparamModule


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
        )

    def forward(self, image, timestep, text):
        time = timestep.reshape(-1, 1, 1, 1) / 1000
        signal = text.mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        prediction = torch.tanh(
            self.weight[0] * image + self.weight[1] * time + self.weight[2] * signal
        )
        return SimpleNamespace(sample=prediction)


class TinyVAE:
    def decode(self, image):
        return SimpleNamespace(sample=torch.tanh(image * 0.1))


from torch import Tensor
import torch.nn.functional as F
from gradcfg.common.optimization import tv_loss
from gradcfg.attacks.base import DEVICE, DEFAULT_NUM_TIMESTEPS


class ReferenceObjective:
    def _compute_gradients(self, params: torch.Tensor) -> Tensor:
        """Compute gradients through unrolled training steps"""
        params.requires_grad_(True)

        # Forward diffusion process
        img = self.dummy_images
        t = self.timesteps * DEFAULT_NUM_TIMESTEPS
        noise = self.gaussian_images
        noisy_img = self.add_noise(img, noise, t)
        text_emb = self.text_emb.expand(self.args.batch_size, -1, -1)

        pred_noise = self.model_init(noisy_img, t, text_emb, flat_param=params).sample
        loss = F.mse_loss(pred_noise, noise)

        # Backpropagation
        grad = torch.autograd.grad(loss, params, create_graph=True)[0]

        return grad

    def reconstruction_loss(self, epoch):
        """Calculate composite reconstruction loss（+ TV 结构先验，可每隔N个epoch计算一次）"""
        init_params = torch.cat(
            [p.reshape(-1) for p in self.model_init.parameters()], 0
        ).detach()
        total_grad = ReferenceObjective._compute_gradients(self, init_params)

        main_loss = 1 - F.cosine_similarity(
            total_grad.flatten(), self.model_grad.flatten(), dim=0, eps=1e-20
        )

        diversity_loss = self._compute_similarity_loss()

        total_loss = (
            main_loss + self.args.ind_lambda * diversity_loss
            if epoch <= self.args.switch_epoch and diversity_loss is not None
            else main_loss
        )

        # ====== TV loss：每隔 tv_every 个 epoch 计算一次 ======
        tv = None
        if (
            self.args.tv_lambda > 0
            and self.args.tv_every > 0
            and (epoch % self.args.tv_every == 0)
        ):
            decoded = self.vae.decode(self.dummy_images / 0.18215).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            tv = tv_loss(decoded)
            total_loss = total_loss + self.args.tv_lambda * tv
        else:
            tv = torch.zeros((), device=DEVICE)  # 方便日志打印
        # ==============================================

        return total_loss, main_loss, diversity_loss, tv


class FastAttackTests(unittest.TestCase):
    def test_tv_defaults_and_explicit_override(self):
        try:
            from gradcfg.attacks.base import parse_args
            from gradcfg.attacks.gradcfg import parse_gradcfg_args as parse_fast_args
        except ImportError as error:
            self.skipTest(f"Diffusion imports unavailable: {error}")
        self.assertEqual(parse_args([]).tv_every, 5)
        self.assertEqual(parse_fast_args([]).tv_every, 5)
        self.assertEqual(parse_fast_args([]).tv_lambda, 0.5)
        overridden = parse_fast_args(["--tv_every", "3", "--tv_lambda", "0.2"])
        self.assertEqual((overridden.tv_every, overridden.tv_lambda), (3, 0.2))

    def test_loss_and_four_variable_gradients_match_original(self):
        try:
            from gradcfg.attacks.base import DiffusionReconstructor as Base
            from gradcfg.attacks.gradcfg import (
                GradCFGReconstructor as FastDiffusionReconstructor,
            )
        except ImportError as error:
            self.skipTest(f"Diffusion imports unavailable: {error}")

        torch.manual_seed(31)
        attack = FastDiffusionReconstructor.__new__(FastDiffusionReconstructor)
        attack.args = SimpleNamespace(
            batch_size=3, tv_lambda=0.2, tv_every=1, switch_epoch=10, ind_lambda=0.03
        )
        attack.dummy_images = torch.randn(
            3, 4, 4, 4, dtype=torch.float64, requires_grad=True
        )
        attack.gaussian_images = torch.randn_like(
            attack.dummy_images, requires_grad=True
        )
        attack.timesteps = torch.full(
            (3,), 0.5, dtype=torch.float64, requires_grad=True
        )
        attack.text_emb = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
        attack.model_init = ReparamModule(TinyUNet())
        attack.attack_parameter = attack.model_init.flat_param.detach().requires_grad_(
            True
        )
        attack.model_grad = torch.randn_like(attack.attack_parameter)
        attack.vae = TinyVAE()
        variables = (
            attack.dummy_images,
            attack.text_emb,
            attack.gaussian_images,
            attack.timesteps,
        )

        expected_loss, _, _, _ = ReferenceObjective.reconstruction_loss(attack, 0)
        expected_grads = torch.autograd.grad(expected_loss, variables)
        actual_loss, _, _, _ = attack.reconstruction_loss(0)
        actual_grads = torch.autograd.grad(actual_loss, variables)

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-10, atol=1e-10)
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        self.assertIsNone(attack.model_init.flat_param.grad)


if __name__ == "__main__":
    unittest.main()
