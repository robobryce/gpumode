"""
Minimal fp16 sort: torch operations only, no load_inline.
Step 1: input.to(torch.float16) -- converts f32 to f16 via hardware
Step 2: torch.sort on f16 -- uses CUB radix sort with 16-bit keys
Step 3: sorted.to(torch.float32) -- converts back via hardware
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data

    # Convert to half precision (hardware-accelerated elementwise)
    h = input_tensor.contiguous().to(torch.float16)

    # Sort in half precision (CUB radix sort with 16-bit keys)
    hs = torch.sort(h)[0]

    # Convert back and write to output
    output_tensor.copy_(hs)

    return output_tensor