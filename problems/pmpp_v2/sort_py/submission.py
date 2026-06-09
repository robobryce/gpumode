"""
Pure Python torch.sort with float32 out=, no int32 view.
Eliminates the int32 view/reinterpret steps. torch.sort on float32
with out= directly into preallocated output buffer.
"""
import torch
from task import input_t, output_t

# Pre-allocate indices buffer for largest benchmark shape
_temp_indices = torch.empty(100_000_000, dtype=torch.int64, device='cuda')


def _get_indices(n: int) -> torch.Tensor:
    return _temp_indices[:n]


def custom_kernel(data: input_t) -> output_t:
    """
    Sort float32 with torch.sort out= directly into output buffer.
    Avoids int32 view overhead. torch.sort is SortPairs internally
    (always computes indices), but out= avoids intermediate allocation.
    """
    input_tensor, output_tensor = data
    indices = _get_indices(input_tensor.numel())
    torch.sort(input_tensor, out=(output_tensor, indices))
    return output_tensor