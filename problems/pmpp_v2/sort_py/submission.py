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
    # Use out= to sort directly into the preallocated output buffer,
    # avoiding intermediate allocation and the fill_reverse_indices /
    # CompareFunctor / elementwise scatter kernels.
    temp_indices = torch.empty_like(data, dtype=torch.int64)
    torch.sort(data, out=(output, temp_indices))
    torch.cuda.synchronize()
    return output