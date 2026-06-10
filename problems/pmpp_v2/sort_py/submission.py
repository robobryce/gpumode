"""
torch.argsort on int32 bitcast + gather.
Tests if argsort uses a different CUB code path than sort.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    int_view = input_tensor.contiguous().view(torch.int32)
    # argsort returns indices — then gather to produce sorted values
    indices = torch.argsort(int_view)
    torch.gather(int_view, 0, indices, out=output_tensor.view(torch.int32))
    return output_tensor
