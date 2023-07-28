import torch
import torch.nn as nn
import numpy as np
import os, argparse, json
from cbctrec.dataPrep import loadDataset
from cbctrec.utils.vis import saveVideo
from cbctrec.eval import Measure_Quality
from PIL import Image
normalize = lambda x: (x-x.min())/(x.max()-x.min())
# tgt_vol_pth = "/storage/Data/cbctrec_data/test-half.npz"
# out_vol_pth = "/storage/NeAT/Experiments/2023-07-17_15-07-08_default/ep0002/volume_test/volume.pt"
__NEAT_HOME = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(__NEAT_HOME, "Experiments", "Eval")

def evalVolume(tgt_vol_pth: str, out_vol_pth: str, out_dir: str = OUT_DIR, save_video: bool = True):
    """Evaluate the volume reconstruction.

    Args:
        tgt_vol_pth (str): Path to the target cbctrec dataset.
        out_vol_pth (str): Path to the output volume.
    """
    print("__________________________________________________________")
    print("Evaluatiing: ", out_vol_pth)
    print("Target: ", tgt_vol_pth)
    _tgt_vol = loadDataset(tgt_vol_pth, "cpu")["volume"]
    tgt_vol = _tgt_vol.transpose(2,0).transpose(2,1).flip(1)
    raw = torch.jit.load(out_vol_pth).state_dict(); assert len(raw) == 1
    out_vol: torch.Tensor = raw["0"].squeeze()

    # out_vol = autoCalibrate(out_vol.cuda(), tgt_vol.cuda())

    tgt_vol_np = tgt_vol.detach().cpu().numpy()
    out_vol_np = out_vol.detach().cpu().numpy()

    assert tgt_vol.shape == out_vol.shape

    psnr = Measure_Quality(tgt_vol_np, out_vol_np, ["PSNRpytorch1"]).item()
    ssim = Measure_Quality(tgt_vol_np, out_vol_np, ["SSIMpytorch"]).item()
    print(f"PSNR/SSIM: {psnr:.3f}/{ssim:.3f}")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # save to json file
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump({
            "tgt_vol_pth": tgt_vol_pth,
            "out_vol_pth": out_vol_pth,
            "PSNR": psnr, 
            "SSIM": ssim
        }, f, indent=4)

    # concatentate tgt and out for visualization
    if save_video:
        vis_out = np.concatenate((tgt_vol_np, out_vol_np), axis=2)
        saveVideo(vis_out, os.path.join(out_dir, "compare.mp4"), fps=10)

    print("Results saved to: ", out_dir)
    
    n_slice= 127
    slice_neat  = normalize(out_vol_np[n_slice])*255  
    slice_neat_image= Image.fromarray(np.uint8(slice_neat))
    slice_neat_image.save(os.path.join(out_dir, "slice_neat"+str(n_slice)+".png"))

def evalProjections(proj_dir, out_dir: str = OUT_DIR, save_video: bool = True):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    proj_pth = None
    tgt_pth = None
    for f in os.listdir(proj_dir):
        if f == "projections.pt":
            proj_pth = os.path.join(proj_dir, f)
        if f == "targets.pt":
            tgt_pth = os.path.join(proj_dir, f)
    assert proj_pth is not None and tgt_pth is not None
    projections = torch.jit.load(proj_pth).state_dict()["0"]
    targets = torch.jit.load(tgt_pth).state_dict()["0"]

    # only use half of the projections
    projections = projections[:projections.shape[0]//2, :, :]
    targets = targets[:targets.shape[0]//2, :, :]

    projections_np = projections.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # TODO: evaluate the projections
    # ...

    # save video
    if save_video:
        print("Saving video to: ", os.path.join(out_dir, "compare_proj.mp4"))
        concat = np.concatenate((projections_np, targets_np), axis=2)
        saveVideo(concat, os.path.join(out_dir, "compare_proj.mp4"), fps=10)


def autoCalibrate(vol: torch.Tensor, aim_vol: torch.Tensor, sample_step: int = 2, apply_shift: bool = False):
    """
    Automatically calibrate the volume to match the aim_vol.
    By:
        1. Find the best shift and scale to match the aim_vol using Gradient Descent.
        2. Apply the shift and scale to the vol.
    """
    class Calibrate(nn.Module):
        def __init__(self):
            super().__init__()
            self.shift = nn.Parameter(torch.zeros(1), requires_grad=apply_shift)
            self.scale = nn.Parameter(torch.ones(1))
        
        def forward(self, x):
            return self.scale * x + self.shift
    
    calibrator = Calibrate().to(vol.device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=1e-3)
    # loss_fn = nn.L1Loss()

    vol_sample = vol[::sample_step, ::sample_step, ::sample_step]
    aim_vol_sample = aim_vol[::sample_step, ::sample_step, ::sample_step]

    class DistanceLoss(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x: torch.Tensor, y: torch.Tensor):
            return (x.mean() - y.mean()).abs() + (x.std() - y.std()).abs()
    loss_fn = DistanceLoss()

    print("Auto-calibrating...")
    __n_epoch = 1000
    for i in range(__n_epoch):
        optimizer.zero_grad()
        loss = loss_fn(calibrator(vol_sample), aim_vol_sample)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"[{int(100*i/__n_epoch):02d}%] Loss: {loss.item():.3f}", end="\r")
    print("Learned shift and scale: ", calibrator.shift.item(), calibrator.scale.item())
    return calibrator(vol).detach()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment directory name or path.")
    # parser.add_argument("--epoch", type=int, required=True, help="Epoch number.")
    parser.add_argument("--ds", type=str, required=True, help="Path to cbctrec dataset.")

    args = parser.parse_args()

    # get last epoch
    exp_dir = os.path.join(__NEAT_HOME, "Experiments", args.exp) if not os.path.isdir(args.exp) else args.exp
    epoch_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d)) and d.startswith("ep")]
    epoch_dirs.sort()
    last_epoch_dir = epoch_dirs[-1]
    
    for dir_pth in epoch_dirs:
        out_vol_pth = os.path.join(dir_pth, "volume", "volume.pt")
        if os.path.exists(out_vol_pth):
            evalVolume(args.ds, out_vol_pth, out_dir=os.path.join(exp_dir, "eval-{}".format(os.path.basename(dir_pth))) , save_video=False)
        proj_dir = os.path.join(dir_pth, "projections")
        if os.path.exists(proj_dir):
            evalProjections(proj_dir, out_dir=os.path.join(exp_dir, "eval-{}".format(os.path.basename(dir_pth))), save_video=False)

    out_vol_pth = os.path.join(last_epoch_dir, "volume", "volume.pt")
    if os.path.exists(out_vol_pth):
        evalVolume(args.ds, out_vol_pth, out_dir=os.path.join(exp_dir, "eval"), save_video=True)
    proj_dir = os.path.join(last_epoch_dir, "projections")
    if os.path.exists(proj_dir):
        evalProjections(proj_dir, out_dir=os.path.join(exp_dir, "eval"), save_video=True)