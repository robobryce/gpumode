"""
Pure-torch sort via torch.sort on int32 bitcast with out=.
ZERO load_inline, ZERO cpp_extension -- pure Python + torch.ops.
torch.sort internally uses CUB Onesweep (same sort kernel as load_inline CUB SortKeys).
No CUDAGraph because torch.sort's internal cudaMalloc for scratch temp storage
is fundamentally incompatible with CUDA graph capture (confirmed by testing:
~50% data corruption on replay regardless of pool/capture_error_mode/stream configs).
This is a known PyTorch limitation.
"""
import torch
from task import input_t, output_t

# Pre-allocate scratch index tensor for torch.sort's out=(values, indices).
# Reuse across calls to avoid per-call allocation overhead.
_max_n = 100_000_000
_scratch_idx = torch.empty(_max_n, dtype=torch.int64, device='cuda')


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via torch.sort on int32 bitcast.
    torch.sort uses CUB DeviceRadixSort internally - same sort kernel as
    the CUB SortKeys load_inline approach, but with SortPairs for indices.
    Pure Python, leaderboard-safe with zero file dependencies.
    """
    input_tensor, output_tensor = data
    n = input_tensor.numel()
    # Sort int32 view directly into output's int32 view.
    torch.sort(input_tensor.view(torch.int32),
               out=(output_tensor.view(torch.int32), _scratch_idx[:n]))
    return output_tensor