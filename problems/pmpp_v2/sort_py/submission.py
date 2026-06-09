"""
Pure torch.sort with int32 bitcast — diagnostic to check if GPU environment
has shifted since brief 2 (parent metric 176.520us). No load_inline, no CUB.
If this matches ~320us (the original baseline), then the regression is real.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    sorted_data, _ = torch.sort(input_tensor)
    output_tensor[:] = sorted_data
    return output_tensor