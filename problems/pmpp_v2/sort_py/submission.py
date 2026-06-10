"""Sort helper — NVRTC radix sort (no nvcc, no load_inline, no cpp_extension)."""
import torch, ctypes
from task import input_t, output_t

# ── CUDA radix sort kernels (self-contained, no CUB) ──────────────────────
_CU_SRC = r"""
#define TILE_SIZE 4096

extern "C" __global__ void radix_hist(const int* __restrict__ in,
                                       int* __restrict__ hist,
                                       int n, int bit) {
    int tid = threadIdx.x, bid = blockIdx.x;
    __shared__ int lh[256];
    for (int i = tid; i < 256; i += 256) lh[i] = 0;
    __syncthreads();
    int s = bid * TILE_SIZE, e = min(s + TILE_SIZE, n);
    for (int i = s + tid; i < e; i += 256) {
        int b = (in[i] >> bit) & 0xFF;
        atomicAdd(&lh[b], 1);
    }
    __syncthreads();
    for (int i = tid; i < 256; i += 256)
        hist[bid * 256 + i] = lh[i];
}

extern "C" __global__ void radix_btot(const int* __restrict__ hist,
                                       int* __restrict__ btot, int nblk) {
    int bin = threadIdx.x;
    if (bin >= 256) return;
    int total = 0;
    for (int b = 0; b < nblk; b++)
        total += hist[b * 256 + bin];
    btot[bin] = total;
}

extern "C" __global__ void radix_goff(const int* __restrict__ btot,
                                       int* __restrict__ goff) {
    if (threadIdx.x > 0) return;
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        goff[i] = sum;
        sum += btot[i];
    }
}

extern "C" __global__ void radix_foffs(const int* __restrict__ hist,
                                        int* __restrict__ offs,
                                        const int* __restrict__ goff,
                                        int nblk) {
    int bin = threadIdx.x;
    if (bin >= 256) return;
    int sum = goff[bin];
    for (int b = 0; b < nblk; b++) {
        offs[b * 256 + bin] = sum;
        sum += hist[b * 256 + bin];
    }
}

extern "C" __global__ void radix_scatter(const int* __restrict__ in,
                                          const int* __restrict__ offs,
                                          int* __restrict__ out,
                                          int n, int bit) {
    int bin = threadIdx.x;
    if (bin >= 256) return;
    int bid = blockIdx.x;
    int base = offs[bid * 256 + bin];
    int s = bid * TILE_SIZE, e = min(s + TILE_SIZE, n);
    // Iterate over tile, pick elements matching this bin, write in order
    for (int i = s; i < e; i++) {
        int key = in[i];
        if (((key >> bit) & 0xFF) == bin) {
            out[base] = key;
            base++;
        }
    }
}
"""

# ── NVRTC and CUDA driver library bindings ────────────────────────────────
_nvrtc = None
_cuda = None

def _load_libs():
    global _nvrtc, _cuda
    if _nvrtc is not None:
        return
    _nvrtc = ctypes.CDLL('libnvrtc.so')
    _cuda = ctypes.CDLL('libcuda.so')
    _cuda.cuInit(0)

def _nvrtc_compile(src_bytes, name, defines=None):
    _load_libs()
    opts = [b'--gpu-architecture=compute_100', b'--std=c++17',
            b'-I/usr/local/cuda/include']
    if defines:
        for d in defines:
            opts.append(f'-D{d}={defines[d]}'.encode())
    opt_arr = (ctypes.c_char_p * len(opts))(*opts)
    prog = ctypes.c_void_p()
    nsrc = src_bytes if isinstance(src_bytes, bytes) else src_bytes.encode()
    res = _nvrtc.nvrtcCreateProgram(ctypes.byref(prog), nsrc, name.encode(),
                                     0, None, None)
    if res != 0:
        raise RuntimeError(f'nvrtcCreateProgram failed: {res}')
    res = _nvrtc.nvrtcCompileProgram(prog, len(opts), opt_arr)
    if res != 0:
        log_sz = ctypes.c_size_t()
        _nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
        log = ctypes.create_string_buffer(log_sz.value)
        _nvrtc.nvrtcGetProgramLog(prog, log)
        _nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f'NVRTC compile error:\n{log.value.decode()}')
    ptx_sz = ctypes.c_size_t()
    _nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
    ptx = ctypes.create_string_buffer(ptx_sz.value)
    _nvrtc.nvrtcGetPTX(prog, ptx)
    _nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.value

_PRIMARY_CTX = None

def _get_ctx():
    global _PRIMARY_CTX
    _load_libs()
    if _PRIMARY_CTX is not None:
        return _PRIMARY_CTX
    torch.cuda.synchronize()
    ctx = ctypes.c_void_p()
    res = _cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    if res == 0 and ctx.value is not None:
        _PRIMARY_CTX = ctx
        return ctx
    res = _cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), ctypes.c_int(0))
    if res == 0:
        _cuda.cuCtxSetCurrent(ctx)
    _PRIMARY_CTX = ctx
    return ctx

def _load_module(ptx_data):
    _load_libs()
    _get_ctx()
    mod = ctypes.c_void_p()
    res = _cuda.cuModuleLoadData(ctypes.byref(mod), ptx_data)
    if res != 0:
        raise RuntimeError(f'cuModuleLoadData failed: {res}')
    return mod

def _get_func(mod, name):
    func = ctypes.c_void_p()
    res = _cuda.cuModuleGetFunction(ctypes.byref(func), mod, name.encode())
    if res != 0:
        raise RuntimeError(f'cuModuleGetFunction({name}) failed: {res}')
    return func

def _launch_kernel(func, grid, block, shmem, in_args):
    """Launch a CUDA kernel via the driver API.
    in_args: list of (is_ptr, value). True=64-bit pointer, False=32-bit int.
    """
    cu64 = ctypes.c_uint64
    ci32 = ctypes.c_int
    cvp = ctypes.c_void_p
    cu = ctypes.c_uint
    holders = []
    ptrs = []
    for is_ptr, val in in_args:
        h = cu64(val) if is_ptr else ci32(val)
        holders.append(h)
        ptrs.append(ctypes.cast(ctypes.pointer(h), cvp))
    arg_arr = (cvp * len(ptrs))(*ptrs)
    res = _cuda.cuLaunchKernel(
        func, cu(grid[0]), cu(grid[1]), cu(grid[2]),
        cu(block[0]), cu(block[1]), cu(block[2]),
        cu(shmem), cvp(0), arg_arr, cvp(0))
    if res != 0:
        raise RuntimeError(f'cuLaunchKernel failed: {res}')

# ── Module cache ───────────────────────────────────────────────────────────
_MODS = {}
_FUNCS = {}
_MAX_N = 0
_DHIST = None
_DOFFS = None
_DBTOT = None
_DGOFF = None
_DTEMP0 = None
_DTEMP1 = None

def _ensure_bufs(n):
    global _MAX_N, _DHIST, _DOFFS, _DBTOT, _DGOFF, _DTEMP0, _DTEMP1
    if n > _MAX_N:
        _MAX_N = n
        if _DTEMP0 is not None:
            del _DHIST, _DOFFS, _DBTOT, _DGOFF, _DTEMP0, _DTEMP1
        nblk = max(1, (n + 4095) // 4096)
        _DHIST = torch.empty(nblk * 256, dtype=torch.int32, device='cuda')
        _DOFFS = torch.empty(nblk * 256, dtype=torch.int32, device='cuda')
        _DBTOT = torch.empty(256, dtype=torch.int32, device='cuda')
        _DGOFF = torch.empty(256, dtype=torch.int32, device='cuda')
        _DTEMP0 = torch.empty(n, dtype=torch.int32, device='cuda')
        _DTEMP1 = torch.empty(n, dtype=torch.int32, device='cuda')

def _get_functions(end_bit):
    _load_libs()
    if end_bit in _MODS:
        return (_MODS[end_bit], _FUNCS[(end_bit, 'hist')],
                _FUNCS[(end_bit, 'btot')], _FUNCS[(end_bit, 'goff')],
                _FUNCS[(end_bit, 'foffs')], _FUNCS[(end_bit, 'scatter')])
    defines = {'END_BIT': str(end_bit)}
    ptx = _nvrtc_compile(_CU_SRC, f'sort_eb{end_bit}.cu', defines)
    mod = _load_module(ptx)
    fh  = _get_func(mod, 'radix_hist')
    fb  = _get_func(mod, 'radix_btot')
    fg  = _get_func(mod, 'radix_goff')
    ff  = _get_func(mod, 'radix_foffs')
    fs  = _get_func(mod, 'radix_scatter')
    _MODS[end_bit] = mod
    for n, f in [('hist',fh), ('btot',fb), ('goff',fg), ('foffs',ff), ('scatter',fs)]:
        _FUNCS[(end_bit, n)] = f
    return mod, fh, fb, fg, ff, fs

def _P(p): return (True, p)             # pointer arg
def _I(v): return (False, int(v))        # int arg

def _sort_keys_radix(in_ptr, out_ptr, n, end_bit):
    _ensure_bufs(n)
    _, fh, fb, fg, ff, fs = _get_functions(end_bit)
    nblk = max(1, (n + 4095) // 4096)
    grid = (nblk, 1, 1)
    block = (256, 1, 1)

    passes = list(range(0, end_bit, 8))
    src_ptr = in_ptr
    dst_tmp0 = True

    for pi, bit in enumerate(passes):
        last = (pi == len(passes) - 1)
        # Histogram
        _launch_kernel(fh, grid, block, 0,
            [_P(src_ptr), _P(_DHIST.data_ptr()), _I(n), _I(bit)])
        # Bin totals
        _launch_kernel(fb, (1,1,1), (256,1,1), 0,
            [_P(_DHIST.data_ptr()), _P(_DBTOT.data_ptr()), _I(nblk)])
        # Global offsets
        _launch_kernel(fg, (1,1,1), (1,1,1), 0,
            [_P(_DBTOT.data_ptr()), _P(_DGOFF.data_ptr())])
        # Final per-block offsets
        _launch_kernel(ff, (1,1,1), (256,1,1), 0,
            [_P(_DHIST.data_ptr()), _P(_DOFFS.data_ptr()),
             _P(_DGOFF.data_ptr()), _I(nblk)])
        # Scatter
        dst = out_ptr if last else (_DTEMP0.data_ptr() if dst_tmp0 else _DTEMP1.data_ptr())
        _launch_kernel(fs, grid, block, 0,
            [_P(src_ptr), _P(_DOFFS.data_ptr()), _P(dst), _I(n), _I(bit)])
        if not last:
            src_ptr = dst
            dst_tmp0 = not dst_tmp0
    torch.cuda.synchronize()

# ── Public API ─────────────────────────────────────────────────────────────

def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    n = input_tensor.numel()
    end_bit = 24 if n <= 10_000_000 else 32
    _sort_keys_radix(input_tensor.data_ptr(), output_tensor.data_ptr(), n, end_bit)
    return output_tensor