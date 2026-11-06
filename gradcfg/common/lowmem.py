"""Exact two-pass gradient matching for a batch-mean training loss."""

import torch
import torch.nn.functional as F


def backward_aggregate_cosine(parameter, observed_gradient, batch_size, sample_loss, attack_variables):
    """Accumulate the gradient of cosine(mean_i grad_theta loss_i, observed).

    sample_loss(index, detached) must return the mean loss for one sample.
    The first pass detaches attack variables; the second pass rebuilds one
    sample's higher-order graph at a time.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if parameter.shape != observed_gradient.shape:
        raise ValueError("Parameter and observed gradient shapes must match")

    aggregate = torch.zeros_like(parameter)
    for index in range(batch_size):
        sample_gradient = torch.autograd.grad(sample_loss(index, True), parameter)[0]
        aggregate.add_(sample_gradient, alpha=1.0 / batch_size)
        del sample_gradient

    aggregate.requires_grad_(True)
    main_loss = 1 - F.cosine_similarity(aggregate, observed_gradient, dim=0, eps=1e-20)
    direction = torch.autograd.grad(main_loss, aggregate)[0].detach()
    direction.div_(batch_size)

    for index in range(batch_size):
        sample_gradient = torch.autograd.grad(
            sample_loss(index, False), parameter, create_graph=True
        )[0]
        torch.autograd.backward(sample_gradient, grad_tensors=direction, inputs=attack_variables)
        del sample_gradient

    return main_loss.detach()
