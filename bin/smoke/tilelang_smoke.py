import tilelang
import tilelang.language as T
import torch


@T.prim_func
def add_one(A: T.Tensor((256,), T.float32), B: T.Tensor((256,), T.float32)):
    with T.Kernel(1, threads=256):
        for i in T.Parallel(256):
            B[i] = A[i] + 1


kernel = tilelang.compile(add_one, out_idx=[1])
x = torch.zeros(256, device="cuda", dtype=torch.float32)
y = kernel(x)
torch.testing.assert_close(y, torch.ones_like(y))
print("TileLang PASS")
