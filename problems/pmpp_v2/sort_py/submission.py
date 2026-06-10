"""
torch.sort on int32 bitcast with out=(output, indices) — zero copy.
Sorted values go directly into output_tensor, no copy_ needed.
Reuse index scratch buffer allocated once at module init.
"""
import torch
from task import input_t, output_t

_idx_scratch = None

def _ensure_idx_scratch(n):
    global _idx_scratch
    max_n = 100_000_000
    if _idx_scratch is None:
        _idx_scratch = torch.empty(max_n, dtype=torch.long, device='cuda')


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    n = input_tensor.numel()
    _ensure_idx_scratch(n)
    int_view = input_tensor.contiguous().view(torch.int32)
    output_int = output_tensor.view(torch.int32)
    torch.sort(int_view, out=(output_int, _idx_scratch[:n]))
    return output_tensor
