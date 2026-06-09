"""
torch.sort on int32 view of float32 data — no load_inline, no CUB, no JIT.
PyTorch internally dispatches to CUB Onesweep for int32, which should match
the parent's 176us approach without any compilation cache issues.
This is the lazy path: let PyTorch's sort JIT handle the dispatch.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    sorted_data, _ = input_tensor.contiguous().view(torch.int32).sort()
    output_tensor.view(torch.int32)[:] = sorted_data
    return output_tensor