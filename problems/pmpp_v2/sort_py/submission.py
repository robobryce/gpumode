"""
Pure-torch sort: torch.sort on int32 bitcast with stable=True.
stable=True might trigger a different CUB dispatch path.
"""
import torch
from task import input_t, output_t

_max_n = 100_000_000
_scratch_idx = torch.empty(_max_n, dtype=torch.int64, device='cuda')


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    n = input_tensor.numel()
    torch.sort(input_tensor.view(torch.int32), stable=True,
               out=(output_tensor.view(torch.int32), _scratch_idx[:n]))
    return output_tensor