
import os, subprocess
from .shared import *

__INTERM_NPZ_DIR = os.path.join(NEAT_HOME, 'scenes', 'interm')
def buildDataset(name: str, path_to_npz: str, cuda_device: int = 0):
    if not os.path.exists(__INTERM_NPZ_DIR):
        os.makedirs(__INTERM_NPZ_DIR)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)

    interm_npz = os.path.join(__INTERM_NPZ_DIR, name + '.npz')
    _run_cmd = ['python', "dataConversion.py", 
                "--name", name.capitalize(), 
                "--ds-path", interm_npz, 
                "--half-range",
                "--yes",
                ]

    if not os.path.exists(interm_npz):
        _run_cmd += ["--ct-path", path_to_npz]
    else:
        print("Interm dataset already exists, skipping re-building...")

    subprocess.run(_run_cmd, **commonRunKwargs())

if __name__ == '__main__':
    # ensure working dir
    assert os.path.basename(os.getcwd()) == 'NeAT'

    # set LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = os.path.join(os.environ['HOME'], 'miniconda3/envs/neat/lib')

    # tasks = []
    # _p = 'p1'; tasks.append(runOnGPU(buildDataset, 0)(_p, SRC_DATASETS[_p]))
    # _p = 'p3'; tasks.append(runOnGPU(buildDataset, 1)(_p, SRC_DATASETS[_p]))
    # _p = 'p4'; tasks.append(runOnGPU(buildDataset, 2)(_p, SRC_DATASETS[_p]))
    # _p = 'p5'; tasks.append(runOnGPU(buildDataset, 3)(_p, SRC_DATASETS[_p]))
    # _p = 'p6'; tasks.append(runOnGPU(buildDataset, 4)(_p, SRC_DATASETS[_p]))
    # _p = 'p7'; tasks.append(runOnGPU(buildDataset, 5)(_p, SRC_DATASETS[_p]))
    # _p = 'p8'; tasks.append(runOnGPU(buildDataset, 6)(_p, SRC_DATASETS[_p]))
    # _p = 'p9'; tasks.append(runOnGPU(buildDataset, 7)(_p, SRC_DATASETS[_p]))
    # _p = 'p10'; tasks.append(runOnGPU(buildDataset, 0)(_p, SRC_DATASETS[_p]))
    # _p = 'p11'; tasks.append(runOnGPU(buildDataset, 1)(_p, SRC_DATASETS[_p]))
    # for t in tasks:
    #     t.join()

    # for _p in SRC_DATASETS.keys():
    #     buildDataset(_p, SRC_DATASETS[_p])
    # startGPUTasks([0, 1, 2, 3, 4, 5, 6, 7])

    buildDataset('p1', SRC_DATASETS['p1'])
    