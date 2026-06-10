"""
Pure torch.sort baseline (no load_inline, no custom kernels).
Measures torch.sort's CUB Onesweep performance directly.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Simple torch.sort — the CUB Onesweep baseline.
    """
    input_tensor, output_tensor = data
    output_tensor[...] = torch.sort(input_tensor)[0]
    return output_tensor