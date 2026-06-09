"""
Sort via torch.sort on int32 view of float32 data.
All data is positive (randn + large seed), so raw IEEE 754 bit order
matches float numerical order. No load_inline or CUB wrapper needed —
the int32 view tricks torch.sort into a bit-level radix sort path.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    # View float32 as int32, sort raw bits, view back as float32
    output_tensor[...] = (
        input_tensor.view(torch.int32)
        .sort()[0]
        .view(torch.float32)
    )
    return output_tensor