
from __future__ import annotations
import os
import multiprocessing
from functools import partial
from typing import Callable

SRC_DATASETS = {
    "p1": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/train_data-p1.npz",
    "p3": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0194247-p3.npz",
    "p4": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0275935-p4.npz",
    "p5": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0280336-p5.npz",
    "p6": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0047120-p6.npz",
    "p7": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0118783-p7.npz",
    "p8": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0389692-p8.npz",
    "p9": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0405910-p9.npz",
    "p10": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0407419-p10.npz",
    "p11": "/remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0424477-p11.npz"
}
def commonRunKwargs():
    return {
        "cwd": NEAT_HOME,
        "env": os.environ.copy(),
        "check": True,
        "shell": False,
    }

NEAT_HOME = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__)))))

def runOnGPU(func, cuda_device: int = 0):
    def wrapper(*args, **kwargs):
        _func = partial(func, cuda_device=cuda_device)
        p = multiprocessing.Process(target=_func, args=args, kwargs=kwargs)
        p.start()
        return p
    return wrapper

GPU_TASKS: list[tuple[Callable, list, dict]] = []
def distributeToGPU(func, args = [], kwargs = {}):
    global GPU_TASKS
    GPU_TASKS.append((func, args, kwargs))

def run_on_gpu(func, cuda_device: int = 0, *args, **kwargs):
    func(*args, **kwargs, cuda_device=cuda_device)
def worker(gpu: int, task_queue: multiprocessing.Queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        func, args, kwargs = task
        run_on_gpu(func, gpu, *args, **kwargs)

def startGPUTasks(gpu_pool: list[int]):
    task_queue = multiprocessing.Queue()
    processes = []

    # Create a worker process for each GPU
    for gpu in gpu_pool:
        p = multiprocessing.Process(target=worker, args=(gpu, task_queue))
        p.start()
        processes.append(p)

    # Add tasks to the queue
    for task in GPU_TASKS:
        task_queue.put(task)

    # Add sentinel tasks to signal the workers to exit
    for _ in range(len(gpu_pool)):
        task_queue.put(None)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Clean up the task queue
    task_queue.close()


__all__ = [
    "NEAT_HOME",
    "SRC_DATASETS",
    "commonRunKwargs",
    "runOnGPU",
    
    "distributeToGPU",
    "startGPUTasks",
]

# if __name__ == "__main__":
#     from cbctrec.dataPrep import makeDataset
#     from cbctrec.config import config
#     makeDataset(
#         "/remote-home/limengxun/tujie/99cab51a7f78ec04ef5b0431f07a6737",
#         SRC_DATASETS["p1"],
#         config
#     )