import torch

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """Correctness-first QR baseline for isolated optimization experiments."""
    return torch.geqrf(data)


def launch_for_eval(inputs: dict) -> output_t:
    return custom_kernel(inputs["data"])


kernel = custom_kernel
