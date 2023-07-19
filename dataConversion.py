"""
Convert data from cbctrec2 to NeAT format
Creates CT_parameters.xtekct file
"""

from cbctrec.conf.loader import ProjectConfig
from cbctrec.dataPrep import loadDataset, makeDataset
import torch, tifffile
import numpy as np
import os, sys, argparse, configparser, shutil

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
    assert config.projection_config["im_shape_HW"][0] == config.projection_config["im_shape_HW"][1], "Only square image supported, not sure about axis order"
    assert config.data_simulation_config["volume_shape"][0] == config.data_simulation_config["volume_shape"][1] == config.data_simulation_config["volume_shape"][2], \
        "Only cubic volume supported, not sure about axis order"
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

TEMPLATE_CONFIG_NAME = "_template.ini"
def createExpConfig(exp_config_dir: str, name: str, split_name: str):
    template_config_file = os.path.join(exp_config_dir, TEMPLATE_CONFIG_NAME)
    __n_proj = split_name[len("exp_uniform_"):]
    dst_config_file = os.path.join(exp_config_dir, f"{name.lower()}_{__n_proj}.ini")

    exp_config = configparser.ConfigParser()
    exp_config.read(template_config_file)
    exp_config["TrainParams"]["name"] = name.lower() + "_" + __n_proj
    exp_config["TrainParams"]["split_name"] = split_name
    exp_config["TrainParams"]["scene_name"] = name.lower()

    # save the config file
    with open(dst_config_file, "w") as f:
        exp_config.write(f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-path", type=str, default=None, help="Path to the CBCT data, if specified, will first generate a cbctrec dataset from the CT data")
    parser.add_argument("--ds-path", type=str, default=None, help="Path to the intermediate cbctrec dataset")
    parser.add_argument("--name", type=str, default="Test", help="Name of the generated NeAT dataset, should be set according to nikon2neat.cpp")
    parser.add_argument("--half-range", action="store_true", help="Whether to use half range data, if the dataset is full range, will only use the first half of the projections")
    args = parser.parse_args()

    NAME: str = args.name
    HALF_RANGE = args.half_range
    # assert capital letter at the beginning
    assert NAME[0].isupper(), "Name must start with a capital letter"
    cbct_path = args.ct_path
    ds_path = args.ds_path
    assert ds_path is not None, "Please specify the path to the intermediate cbctrec dataset"

    # cbct_path = "/storage/Data/cbctrec_data/99cab51a7f78ec04ef5b0431f07a6737"
    # ds_path = "/storage/Data/cbctrec_data/test-half.npz"

    # NeAT somehow requires full circular data...
    # if the dataset is half range, we need to double the number of projections to adapt to NeAT
    if cbct_path is not None:
        from cbctrec.config import config
        if not config.data_simulation_config["full_range"]:
            config.data_simulation_config["full_range"] = True
            config.data_simulation_config["n_total_projection"] *= 2
            print("Compromising half range data to full range data")
            assert HALF_RANGE, "If the dataset is half range, please specify --half-range"

        print(f"Making dataset with {'full' if config.data_simulation_config['full_range'] else 'half'} range {config.data_simulation_config['n_total_projection']} projections")
        makeDataset(cbct_path, ds_path, config)

    ds = loadDataset(ds_path)
    assert ds["config"].data_simulation_config["full_range"] == True, "Dataset must be full range!"

    # Aim paths
    __NEAT_HOME = os.path.dirname(os.path.abspath(__file__))
    nikon_dst_dir = os.path.join(__NEAT_HOME, "scenes", NAME)           # save nikon version of the dataset
    neat_dst_dir = os.path.join(__NEAT_HOME, "scenes", NAME.lower())    # save neat version of the dataset
    if os.path.exists(nikon_dst_dir) or os.path.exists(neat_dst_dir):
        if input(f"Dataset {NAME} already exists, overwrite? (y/n)") == "y":
            if os.path.exists(nikon_dst_dir):
                shutil.rmtree(nikon_dst_dir)
                print(f"Removed {nikon_dst_dir}")
            if os.path.exists(neat_dst_dir):
                shutil.rmtree(neat_dst_dir)
                print(f"Removed {neat_dst_dir}")
        else:
            print(f"Dataset {NAME} already exists, aborting")
            exit()

    exp_config_dir = os.path.join(__NEAT_HOME, "configs")
    img_dst_dir = os.path.join(nikon_dst_dir, "projections")  # save tiff images
    for _pth in [nikon_dst_dir, img_dst_dir, neat_dst_dir]:
        if not os.path.exists(_pth):
            os.mkdir(_pth)

    # save xtekct config
    xtekct_config = cvtConfig(ds["config"], NAME)
    toXtekctFile(xtekct_config, os.path.join(nikon_dst_dir, f"{NAME}_CT_parameters.xtekct"))

    # save volume
    _gt_volume = ds["volume"].transpose(2,0).transpose(2,1).flip(1).unsqueeze(0)
    assert _gt_volume.shape == (1, 256, 256, 256), "Volume shape must be set according to SceneBase.cpp"
    _gt_volume.detach().cpu().to(torch.float32).contiguous().numpy().tofile(os.path.join(nikon_dst_dir, f"volume_gt.bin"))

    images: list[torch.Tensor] = ds["projections"]
    images_tensor = torch.stack(images, dim=0)

    MAX_VAL = 64000     # White level?
    images_tensor = 1 - images_tensor
    images_tensor = (images_tensor - images_tensor.min()) / (images_tensor.max() - images_tensor.min())

    # images_tensor = (1-images_tensor) * MAX_VAL
    images_tensor = images_tensor * MAX_VAL
    images_tensor = images_tensor.to(torch.int16)

    # save images as tiff
    if not os.path.exists(img_dst_dir):
        os.mkdir(img_dst_dir)
    for i, img in enumerate(images_tensor):
        D_TYPE = np.uint16
        tifffile.imwrite(f"{img_dst_dir}/{NAME}_{i:04d}.tif", img.cpu().numpy().astype(D_TYPE), dtype=D_TYPE)
    
    # split dataset
    if not os.path.exists(neat_dst_dir):
        os.mkdir(neat_dst_dir)
    exp_n_images = [15, 20, 30, 40, 60]
    n_total_projection = ds["config"].data_simulation_config["n_total_projection"]

    if HALF_RANGE:
        __n_total_projection = n_total_projection//2
    else:
        __n_total_projection = n_total_projection
    
    if __n_total_projection >= 2*exp_n_images[-1]:
        exp_n_images.append(2*exp_n_images[-1])

    exp_dir_names = [f"exp_uniform_{i}" for i in exp_n_images]
    for _n in exp_n_images:
        assert __n_total_projection % _n == 0 or __n_total_projection // _n > 2
    for exp_dir_name, n_image in zip(exp_dir_names, exp_n_images):
        exp_dir = os.path.join(neat_dst_dir, exp_dir_name)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        # create exp config
        train_ims = []
        eval_ims = []
        step = __n_total_projection / n_image

        __n = 0
        for i in range(__n_total_projection):
            if i < __n:
                eval_ims.append(i)
            else:
                train_ims.append(i)
                __n += step

        with open(os.path.join(exp_dir, "train.txt"), "w") as f:
            for i in train_ims:
                f.write(f"{i}\n")
        with open(os.path.join(exp_dir, "eval.txt"), "w") as f:
            for i in eval_ims:
                f.write(f"{i}\n")
        print(f"Created {exp_dir_name} with {len(train_ims)} training images and {len(eval_ims)} evaluation images")
    
    # create exp config
    for n_image in exp_n_images:
        createExpConfig(exp_config_dir, NAME, f"exp_uniform_{n_image}")

    print("Done!")
