import torch
import numpy as np
import os, argparse, json
from cbctrec.dataPrep import loadDataset
from cbctrec.utils.vis import saveVideo
from cbctrec.eval import Measure_Quality

# tgt_vol_pth = "/storage/Data/cbctrec_data/test-half.npz"
# out_vol_pth = "/storage/NeAT/Experiments/2023-07-17_15-07-08_default/ep0002/volume_test/volume.pt"
__NEAT_HOME = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(__NEAT_HOME, "Experiments", "Eval")

def evalVolume(tgt_vol_pth: str, out_vol_pth: str):
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

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    
    tgt_vol_np = normalize(tgt_vol).detach().cpu().numpy()
    out_vol_np = normalize(out_vol).detach().cpu().numpy()

    # print(tgt_vol.shape)
    # print(out_vol.shape)
    # import pdb; pdb.set_trace()
    assert tgt_vol.shape == out_vol.shape

    psnr = Measure_Quality(tgt_vol_np, out_vol_np, ["PSNRpytorch1"]).item()
    ssim = Measure_Quality(tgt_vol_np, out_vol_np, ["SSIMpytorch"]).item()
    print(f"PSNR/SSIM: {psnr:.3f}/{ssim:.3f}")

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    # save to json file
    with open(os.path.join(OUT_DIR, "eval.json"), "w") as f:
        json.dump({"PSNR": psnr, "SSIM": ssim}, f)

    # concatentate tgt and out for visualization
    vis_out = np.concatenate((tgt_vol_np, out_vol_np), axis=2)
    saveVideo(vis_out, os.path.join(OUT_DIR, "eval.mp4"), fps=10)

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

    # out_vol_pth = os.path.join(__NEAT_HOME, "Experiments", args.exp, f"ep{args.epoch:04d}", "volume_test", "volume.pt")
    out_vol_pth = os.path.join(last_epoch_dir, "volume_test", "volume.pt")
    evalVolume(args.ds, out_vol_pth)