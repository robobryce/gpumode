import os, sys, multiprocessing

# SPAWN context inherits env, set here before any multiprocessing
os.environ['PYTHONPATH'] = '/home/shadeform/gpumode/autocuda/worktrees/optimize-2026-06-09-06-58-37-worker-2/problems/pmpp_v2/sort_py:/home/shadeform/gpumode/autocuda/worktrees/optimize-2026-06-09-06-58-37-worker-2/problems/pmpp_v2'
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def spawn_check():
    import sys, os
    for i, p in enumerate(sys.path[:15]):
        print(f'SPAWN sys.path[{i}] = {p}', flush=True)
    print(f'SPAWN PYTHONPATH={os.environ.get("PYTHONPATH","NOT SET")}', flush=True)
    import submission
    print(f'SPAWN submission.__file__={submission.__file__}', flush=True)
    print(f'SPAWN has _custom_kernel={hasattr(submission, "_custom_kernel")}', flush=True)
    print(f'SPAWN has sort_module={hasattr(submission, "sort_module")}', flush=True)
    return submission.__file__

if __name__ == '__main__':
    print(f'PARENT sys.path[:7] = {sys.path[:7]}', flush=True)
    import submission as sub_parent
    print(f'PARENT submission.__file__={sub_parent.__file__}', flush=True)
    print(f'PARENT has _custom_kernel={hasattr(sub_parent, "_custom_kernel")}', flush=True)
    
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(1) as pool:
        result = pool.apply(spawn_check)
        print(f'RESULT: {result}', flush=True)
