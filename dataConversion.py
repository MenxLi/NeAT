"""
Convert data from cbctrec2 to NeAT format
Creates CT_parameters.xtekct file
"""

from cbctrec.conf.loader import ProjectConfig
from cbctrec.dataPrep import loadDataset, makeDataset
import torch, tifffile
import numpy as np
import matplotlib.pyplot as plt
import os, sys

def cvtConfig(config: ProjectConfig, name: str):
    # taken from example xtekct file
    xtekct_config = dict(
        Name=name,
        VoxelsX=326,
        VoxelsY=345,
        VoxelsZ=245,
        VoxelSizeX=0.427,
        VoxelSizeY=0.427,
        VoxelSizeZ=0.427,
        OffsetX=-4.0122,
        OffsetY=-0.0127,
        OffsetZ=1.6803,
        SrcToObject=595.033142,
        SrcToDetector=983.0,
        MaskRadius=73.0896,
        DetectorPixelsX=1916,
        DetectorPixelsY=1536,
        DetectorPixelSizeX=0.127,
        DetectorPixelSizeY=0.127,
        DetectorOffsetX=0.0,
        DetectorOffsetY=0.0,
        CentreOfRotation=0.0,
        CentreOfRotationTop=0.0,
        CentreOfRotationBottom=0.0,
        WhiteLevel=64000.0,
        Scattering=0.0,
        CoefX4=0.0,
        CoefX3=0.0,
        CoefX2=0.0,
        CoefX1=1.0,
        CoefX0=0.0,
        Scale=1.0,
        RegionStartX=0,
        RegionStartY=0,
        RegionPixelsX=1916,
        RegionPixelsY=1536,
        Projections=360,
        InitialAngle=0.0,
        AngularStep=1.0,
        FilterType=0,
        CutOffFrequency=3.937008,
        Exponent=1.0,
        Normalisation=1.0,
        InterpolationType=1,
        Scaling=1000.0,
        OutputType=0,
        ImportConversion=1,
        TIFFScaling=1,
        TimeStampFolder=1,
    )

    _src_to_detector = config.data_simulation_config["camera_distance"] * 2
    _angular_step = 360 / config.data_simulation_config["n_total_projection"] if config.data_simulation_config["full_range"] else \
        180 / config.data_simulation_config["n_total_projection"]
    assert _angular_step % 1 == 0, "Angular step ...somehow... must be integer"
    update_config = dict(
        Name = "cbctrec",
        VoxelsX = config.data_simulation_config["volume_shape"][0],
        VoxelsY = config.data_simulation_config["volume_shape"][1],
        VoxelsZ = config.data_simulation_config["volume_shape"][1],

        VoxelSizeX = config.projection_config["voxel_size"],
        VoxelSizeY = config.projection_config["voxel_size"],
        VoxelSizeZ = config.projection_config["voxel_size"],

        # Offsets [Unsure]
        OffsetX = 0,
        OffsetY = 0,
        OffsetZ = 0,

        SrcToObject = config.data_simulation_config["camera_distance"],
        SrcToDetector = _src_to_detector,

        # MaskRadius ?
        # MaskRadius = 0,

        DetectorPixelsX = config.projection_config["im_shape_HW"][0],
        DetectorPixelsY = config.projection_config["im_shape_HW"][1],
        DetectorPixelSizeX = config.projection_config["pixel_size"] * _src_to_detector / config.projection_config["focal_len"],
        DetectorPixelSizeY = config.projection_config["pixel_size"] * _src_to_detector / config.projection_config["focal_len"],

        # RegionPixels [Unsure]
        RegionPixelsX = config.projection_config["im_shape_HW"][0],
        RegionPixelsY = config.projection_config["im_shape_HW"][1],

        Projections = config.data_simulation_config["n_total_projection"],
        AngularStep = _angular_step
    )

    xtekct_config.update(update_config)

    return xtekct_config

__tail = """
[Xrays]
XraykV=105
XrayuA=727

[CTPro]
Version=V2.2.3644.20140 (Date:23 December 2009)
Product=Product:[CT Pro], SuitBuild:[Metris XT, Build 2.0], Description:[] Copyright:[Copyright ? Metris 2004 ? 2015]
Shuttling=True
ROR_Auto=True
Filter_ThicknessMM=
Filter_Material=None
BeamHardening_Type=Simple
BeamHardening_Simple=1
AutoCOR_NumBands=1
AutoCOR_Accuracy=0.1
SliceSingle_HeightPx=768
SliceSingle_Region={X=0,Y=0,Width=0,Height=0}
SliceDualTop_HeightPx=1152
SliceDualTop_Region={X=0,Y=0,Width=0,Height=0}
SliceDualBottom_HeightPx=384
SliceDualBottom_Region={X=0,Y=0,Width=0,Height=0}
Scaling_DataTypeVolume=Float
Scaling_Units=InverseMetres
Scaling_CustomValue=1.0
Calibration_Calc_Value=1
Calibration_Calc_Material=Water
Calibration_Calc_RelativeDensity=1
Calibration_Calc_DiameterMM=0
Calibration_Calc_Region={X=0,Y=0,Width=0,Height=0}
Calibration_Calc_File=
AngleFile_Use=True
AngleFile_IgnoreErrors=True
"""
def toXtekctFile(xtekct_config: dict, save_path: str):
    with open(save_path, "w") as f:
        f.write("[XtekCT]\n")
        for key, value in xtekct_config.items():
            f.write(f"{key}={value}\n")
        f.write(__tail)

if __name__ == "__main__":
    cbct_path = "/storage/Data/cbctrec_data/99cab51a7f78ec04ef5b0431f07a6737"
    ds_path = "/storage/Data/cbctrec_data/test-half.npz"
    # ds_path = "/storage/Data/cbctrec_data/test.npz"
    # ds_path = "/storage/Data/cbctrec_data/test-half-120.npz"
    # ds_path = "/storage/Data/cbctrec_data/test-half-360.npz"

    # print("Making dataset...")
    # from cbctrec.config import config
    # config.data_simulation_config["full_range"] = False
    # config.data_simulation_config["n_total_projection"] = 180
    # makeDataset(cbct_path, ds_path, config)

    ds = loadDataset(ds_path)

    # create dst dir
    NAME = "Test"
    assert NAME[0].isupper()     # assert capital letter at the beginning
    dst_dir = f"/storage/NeAT/scenes/{NAME}"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # save xtekct config
    xtekct_config = cvtConfig(ds["config"], NAME)
    toXtekctFile(xtekct_config, f"{dst_dir}/{NAME}_CT_parameters.xtekct")

    images: list[torch.Tensor] = ds["projections"]
    images_tensor = torch.stack(images, dim=0)
    # normalize images
    images_tensor = (images_tensor - images_tensor.min()) / (images_tensor.max() - images_tensor.min())
    images_tensor = 1 - images_tensor
    MAX_VAL = 64000     # White level?
    MIN_VAL = 1000
    images_tensor = images_tensor * (MAX_VAL-MIN_VAL) + MIN_VAL
    images_tensor = images_tensor.to(torch.int16)

    # save images as tiff
    img_dst_dir = f"{dst_dir}/projections"
    if not os.path.exists(img_dst_dir):
        os.mkdir(img_dst_dir)
    for i, img in enumerate(images_tensor):
        # save as unsigned 8-bit integer
        D_TYPE = np.uint16
        tifffile.imwrite(f"{img_dst_dir}/{NAME}_{i:04d}.tif", img.cpu().numpy().astype(D_TYPE), dtype=D_TYPE)
    
    # create exp file
    neat_dst_dir = f"/storage/NeAT/scenes/{NAME.lower()}"
    if not os.path.exists(neat_dst_dir):
        os.mkdir(neat_dst_dir)
    exp_n_images = [15, 20, 30, 40, 60]
    exp_dir_names = [f"exp_uniform_{i}" for i in exp_n_images]
    n_total_projection = ds["config"].data_simulation_config["n_total_projection"]
    for _n in exp_n_images:
        assert n_total_projection % _n == 0 or n_total_projection // _n > 2
    for exp_dir_name, n_image in zip(exp_dir_names, exp_n_images):
        exp_dir = f"{neat_dst_dir}/{exp_dir_name}"
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        # create exp config
        train_ims = []
        eval_ims = []
        step = n_total_projection / n_image

        __n = 0
        for i in range(n_total_projection):
            if i < __n:
                eval_ims.append(i)
            else:
                train_ims.append(i)
                __n += step

        with open(f"{exp_dir}/train.txt", "w") as f:
            for i in train_ims:
                f.write(f"{i}\n")
        with open(f"{exp_dir}/eval.txt", "w") as f:
            for i in eval_ims:
                f.write(f"{i}\n")




