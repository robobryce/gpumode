import torch
import triton
import triton.language as tl


@triton.jit
def add_one(x, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    values = tl.load(x + offsets, mask=offsets < n)
    tl.store(x + offsets, values + 1, mask=offsets < n)


x = torch.zeros(256, device="cuda", dtype=torch.float32)
add_one[(1,)](x, x.numel(), block=256)
torch.testing.assert_close(x, torch.ones_like(x))
print("Triton PASS")
