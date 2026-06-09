import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Implements sort using PyTorch.
    Args:
        data: Input tensor to be sorted
    Returns:
        Sorted tensor
    """
    data, output = data
    output[...] = torch.sort(data)[0]
    torch.cuda.synchronize()
    return output