"""Weight change statistics shared by both fine-tuning workflows."""
import torch


def calculate_weight_changes(initial_weights, final_weights):
    delta_dict = {}
    for key in initial_weights:
        delta_dict[key] = final_weights[key] - initial_weights[key]

    def flatten_weights(weights):
        flattened = []
        for key in weights:
            flattened.append(weights[key].flatten())
        return torch.cat(flattened)

    initial_vec = flatten_weights(initial_weights)
    delta_vec = flatten_weights(delta_dict)

    norm_initial = torch.norm(initial_vec, p=2).item()
    norm_delta = torch.norm(delta_vec, p=2).item()
    ratio = norm_delta / norm_initial if norm_initial > 0 else 0

    return {
        "norm_initial": norm_initial,
        "norm_delta": norm_delta,
        "delta_ratio": ratio
    }
