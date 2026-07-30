import torch
import cuda.tile as ct


@ct.kernel()
def add_one(x, n: ct.Constant[int]):
    tile = ct.load(x, index=(ct.bid(0),), shape=(n,))
    ct.store(x, index=(ct.bid(0),), tile=tile + 1)


x = torch.zeros(256, device="cuda", dtype=torch.float32)
ct.launch(torch.cuda.current_stream(), (1,), add_one, (x, x.numel()))
torch.cuda.synchronize()
torch.testing.assert_close(x, torch.ones_like(x))
print("cuTile Python PASS")
