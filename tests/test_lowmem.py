"""Numerical checks for exact batch gradient matching without diffusion weights."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from gradcfg.common.lowmem import backward_aggregate_cosine
from gradcfg.common.reparam import ReparamModule


class TwoPassGradientTests(unittest.TestCase):
    def test_four_attack_variables_match_full_batch(self):
        torch.manual_seed(13)
        batch_size = 5
        dtype = torch.float64
        parameter = torch.randn(16, dtype=dtype, requires_grad=True)
        observed = torch.randn_like(parameter)
        image = torch.randn(batch_size, 3, dtype=dtype, requires_grad=True)
        noise = torch.randn(batch_size, 3, dtype=dtype, requires_grad=True)
        timestep = torch.rand(batch_size, dtype=dtype, requires_grad=True)
        text = torch.randn(1, 2, dtype=dtype, requires_grad=True)
        variables = (image, text, noise, timestep)

        def sample_loss(index, detached=False):
            x, n, t, c = image[index], noise[index], timestep[index], text[0]
            if detached:
                x, n, t, c = (item.detach() for item in (x, n, t, c))
            hidden = torch.tanh(F.linear(x + n * t, parameter[:12].reshape(4, 3), parameter[12:]))
            prediction = hidden[:2] + c
            return (prediction - n[:2]).square().mean()

        full_training_loss = torch.stack([sample_loss(i) for i in range(batch_size)]).mean()
        full_gradient = torch.autograd.grad(full_training_loss, parameter, create_graph=True)[0]
        full_cosine = 1 - F.cosine_similarity(full_gradient, observed, dim=0, eps=1e-20)
        expected = torch.autograd.grad(full_cosine, variables)

        actual_cosine = backward_aggregate_cosine(
            parameter, observed, batch_size, sample_loss, variables
        )
        torch.testing.assert_close(actual_cosine, full_cosine.detach(), rtol=1e-12, atol=1e-12)
        for variable, gradient in zip(variables, expected):
            torch.testing.assert_close(variable.grad, gradient, rtol=1e-10, atol=1e-10)
        self.assertIsNone(parameter.grad)

    def test_chunked_tv_matches_batch_mean(self):
        from gradcfg.common.optimization import tv_loss

        torch.manual_seed(17)
        image = torch.randn(5, 3, 6, 6, dtype=torch.float64, requires_grad=True)
        expected = torch.autograd.grad(tv_loss(image), image)[0]
        for index in range(image.shape[0]):
            (tv_loss(image[index:index + 1]) / image.shape[0]).backward()
        torch.testing.assert_close(image.grad, expected, rtol=1e-12, atol=1e-12)

    def test_reparameterized_model_matches_full_batch(self):
        torch.manual_seed(19)
        model = ReparamModule(torch.nn.Sequential(
            torch.nn.Linear(3, 5, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 3, dtype=torch.float64),
        ))
        parameter = model.flat_param.detach().requires_grad_(True)
        images = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        observed = torch.randn_like(parameter)

        def sample_loss(index, detached=False):
            sample = images[index:index + 1]
            if detached:
                sample = sample.detach()
            prediction = model(sample, flat_param=parameter)
            return (prediction - sample).square().mean()

        loss = torch.stack([sample_loss(i) for i in range(3)]).mean()
        gradient = torch.autograd.grad(loss, parameter, create_graph=True)[0]
        full_cosine = 1 - F.cosine_similarity(gradient, observed, dim=0, eps=1e-20)
        expected = torch.autograd.grad(full_cosine, images)[0]

        backward_aggregate_cosine(parameter, observed, 3, sample_loss, (images,))
        torch.testing.assert_close(images.grad, expected, rtol=1e-10, atol=1e-10)
        self.assertIsNone(model.flat_param.grad)

    def test_full_lowmem_step_matches_original_objective(self):
        try:
            from gradcfg.attacks.lowmem import LowMemoryDiffusionReconstructor
        except ImportError as error:
            self.skipTest(f"Diffusion imports unavailable: {error}")
        from gradcfg.common.optimization import tv_loss

        class TinyUNet:
            def __call__(self, image, timestep, text, flat_param):
                signal = text.mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
                time = timestep.reshape(-1, 1, 1, 1) / 1000
                prediction = torch.tanh(
                    flat_param[0] * image + flat_param[1] * time + flat_param[2] * signal
                )
                return SimpleNamespace(sample=prediction)

        class TinyVAE:
            def decode(self, image):
                return SimpleNamespace(sample=torch.tanh(image * 0.1))

        torch.manual_seed(29)
        reconstructor = LowMemoryDiffusionReconstructor.__new__(LowMemoryDiffusionReconstructor)
        reconstructor.args = SimpleNamespace(batch_size=3, tv_lambda=0.2, tv_every=1,
                                             switch_epoch=10, ind_lambda=0.03)
        reconstructor.dummy_images = torch.randn(3, 4, 4, 4, dtype=torch.float64, requires_grad=True)
        reconstructor.gaussian_images = torch.randn_like(reconstructor.dummy_images, requires_grad=True)
        reconstructor.timesteps = torch.full((3,), 0.5, dtype=torch.float64, requires_grad=True)
        reconstructor.text_emb = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
        reconstructor.attack_parameter = torch.randn(3, dtype=torch.float64, requires_grad=True)
        reconstructor.model_grad = torch.randn(3, dtype=torch.float64)
        reconstructor.model_init = TinyUNet()
        reconstructor.vae = TinyVAE()
        variables = (reconstructor.dummy_images, reconstructor.text_emb,
                     reconstructor.gaussian_images, reconstructor.timesteps)

        time = reconstructor.timesteps * 1000
        noisy = reconstructor.add_noise(reconstructor.dummy_images,
                                        reconstructor.gaussian_images, time)
        prediction = reconstructor.model_init(
            noisy, time, reconstructor.text_emb.expand(3, -1, -1),
            flat_param=reconstructor.attack_parameter,
        ).sample
        training_loss = F.mse_loss(prediction, reconstructor.gaussian_images)
        aggregate = torch.autograd.grad(training_loss, reconstructor.attack_parameter,
                                        create_graph=True)[0]
        main = 1 - F.cosine_similarity(aggregate, reconstructor.model_grad,
                                       dim=0, eps=1e-20)
        decoded = reconstructor.vae.decode(reconstructor.dummy_images / 0.18215).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        full_loss = main + 0.2 * tv_loss(decoded) + 0.03 * reconstructor._compute_similarity_loss()
        expected = torch.autograd.grad(full_loss, variables)

        actual_loss, _, _, _ = reconstructor._backward_step(0)
        torch.testing.assert_close(actual_loss, full_loss.detach(), rtol=1e-10, atol=1e-10)
        for variable, gradient in zip(variables, expected):
            torch.testing.assert_close(variable.grad, gradient, rtol=1e-9, atol=1e-9)
        self.assertIsNone(reconstructor.attack_parameter.grad)


if __name__ == "__main__":
    unittest.main()
