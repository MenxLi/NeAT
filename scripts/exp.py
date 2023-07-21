
import os, subprocess
from .shared import *

def runExperiment(name: str, n_views: int, cuda_device: int = 0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    subprocess.run(['./build/bin/nikon2neat', name.capitalize()], **commonRunKwargs())
    subprocess.run(["./build/bin/reconstruct", f"configs/exp/{name}_{n_views}.ini"])

if __name__ == '__main__':
    # ensure working dir
    assert os.path.basename(os.getcwd()) == 'NeAT'

    # set LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = os.path.join(os.environ['HOME'], 'miniconda3/envs/neat/lib')

    def distributeInstance(patient_id: str):
        distributeToGPU(runExperiment, [patient_id, 10])
        distributeToGPU(runExperiment, [patient_id, 15])
        distributeToGPU(runExperiment, [patient_id, 20])
        distributeToGPU(runExperiment, [patient_id, 30])
        distributeToGPU(runExperiment, [patient_id, 40])
        distributeToGPU(runExperiment, [patient_id, 60])
    
    distributeInstance('p6')
    distributeInstance('p7')
    distributeInstance('p8')
    distributeInstance('p9')
    distributeInstance('p10')
    distributeInstance('p11')
    startGPUTasks([0, 1, 2, 3, 4, 5, 6, 7])
