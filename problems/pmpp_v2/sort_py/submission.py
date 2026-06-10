"""
Pure PyTorch sort -- exact same pattern as reference.
No load_inline, no bitcast, no view tricks.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    data_tensor, output_tensor = data
    output_tensor[...] = torch.sort(data_tensor)[0]
    return output_tensor
