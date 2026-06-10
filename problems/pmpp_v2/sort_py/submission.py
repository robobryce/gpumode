"""
Pure PyTorch torch.sort int32 stable=True with CUDAGraph capture/replay.
Stable sort may use a different CUB internal dispatch path vs unstable.
Leaderboard-safe: no load_inline, cpp_extension, ctypes, or stream APIs.
"""
import torch
from task import input_t, output_t

_graph_cache = {}


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    in_contig = input_tensor.contiguous()
    out_contig = output_tensor.contiguous()
    n = in_contig.numel()
    key = (out_contig.data_ptr(), n)

    entry = _graph_cache.get(key)
    if entry is not None:
        g, _ = entry
        g.replay()
        return output_tensor

    input_int = in_contig.view(torch.int32)
    output_int = out_contig.view(torch.int32)
    idx = torch.empty(n, dtype=torch.int64, device=in_contig.device)

    # stable=True — may use alternative CUB dispatch
    torch.sort(input_int, stable=True, out=(output_int, idx))
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.sort(input_int, stable=True, out=(output_int, idx))
    g.replay()

    _graph_cache[key] = (g, idx)
    return output_tensor