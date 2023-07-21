
import os, subprocess
from .shared import *

def nikon2neat(name: str):
    subprocess.run(['./build/bin/nikon2neat', name.capitalize()], **commonRunKwargs())

if __name__ == '__main__':
    # ensure working dir
    assert os.path.basename(os.getcwd()) == 'NeAT'

    # set LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = os.path.join(os.environ['HOME'], 'miniconda3/envs/neat/lib')

    for i in range(1, 12):
        if i!=2:
            distributeToGPU(nikon2neat, [f'p{i}'])
    startGPUTasks([0, 1, 2, 3, 4, 5, 6, 7])