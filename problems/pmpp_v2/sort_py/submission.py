"""
Pure PyTorch torch.sort on int32 bitcast -- NO CUDAGraph, no streams, no custom CUDA.
Leaderboard-safe: only torch.sort and .view() operations, default stream.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data

    in_int = input_tensor.contiguous().view(torch.int32)
    out_int = output_tensor.contiguous().view(torch.int32)

    indices = torch.empty(input_tensor.numel(), dtype=torch.int64, device=input_tensor.device)
    torch.sort(in_int, out=(out_int, indices))

    return output_tensor