import os, sys, multiprocessing

os.environ['PYTHONPATH'] = '/home/shadeform/gpumode/autocuda/worktrees/optimize-2026-06-09-06-58-37-worker-2/problems/pmpp_v2/sort_py:/home/shadeform/gpumode/autocuda/worktrees/optimize-2026-06-09-06-58-37-worker-2/problems/pmpp_v2'
os.environ['PYTHONNOUSERSITE'] = '1'

def test_fn():
    print('SPAWNED sys.path[:7]:', sys.path[:7], flush=True)
    import submission
    print('SPAWNED submission.__file__:', submission.__file__, flush=True)
    print('SPAWNED has _custom_kernel:', hasattr(submission, '_custom_kernel'), flush=True)
    return submission.__file__

if __name__ == '__main__':
    import submission as sub_parent
    print('PARENT submission.__file__:', sub_parent.__file__, flush=True)
    print('PARENT has _custom_kernel:', hasattr(sub_parent, '_custom_kernel'), flush=True)
    
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(1) as pool:
        result = pool.apply(test_fn)
        print('PARENT result:', result, flush=True)
