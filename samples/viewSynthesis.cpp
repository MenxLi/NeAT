/*
DEPRECATED, 
the model weights somehow cannot be loaded correctly,
*/


#include "saiga/core/image/freeimage.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ColorizeTensor.h"
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/TrainParameters.h"

#include "Settings.h"
#include "data/Dataloader.h"
#include "data/SceneBase.h"
#include "structure/HyperTree.h"
#include "utils/utils.h"
#include <iostream>
#include <memory>

#include "build_config.h"
#include "geometry/geometry_ex_ex.h"
#include "tensorboard_logger.h"
#include "utils/cimg_wrapper.h"
using namespace Saiga;


struct TrainScene
{
    std::shared_ptr<SceneBase> scene;
    HyperTreeBase tree = nullptr;
    std::shared_ptr<HierarchicalNeuralGeometry> neural_geometry;

    double last_eval_loss           = 9237643867809436;
    double new_eval_loss            = 9237643867809436;
    int last_structure_change_epoch = 0;

    void SaveCheckpoint(const std::string& dir)
    {
        auto prefix = dir + "/" + scene->scene_name + "_";
        scene->SaveCheckpoint(dir);

        torch::nn::ModuleHolder<HierarchicalNeuralGeometry> holder(neural_geometry);
        torch::save(holder, prefix + "geometry.pth");

        torch::save(tree, prefix + "tree.pth");
    }

    void LoadCheckpoint(const std::string& dir)
    {
        std::cout << "Load checkpoint " << dir << std::endl;
        auto prefix = dir + "/" + scene->scene_name + "_";
        // scene->LoadCheckpoint(dir);
        std::cout << "Load scene done" << std::endl;

        if (std::filesystem::exists(prefix + "tree.pth"))
        {
            std::cout << "Load checkpoint tree " << std::endl;
            torch::load(tree, prefix + "tree.pth");
        }
        if (std::filesystem::exists(prefix + "geometry.pth"))
        {
            std::cout << "Load checkpoint geometry " << std::endl;
            torch::nn::ModuleHolder<HierarchicalNeuralGeometry> holder(neural_geometry);
            torch::load(holder, prefix + "geometry.pth");
        }
    }
};

TrainScene initializeScene(TrainScene& ts, std::shared_ptr<CombinedParams> params)
{
    assert(params->train_params.scene_name.size() == 1);
    std::basic_string<char> scene_name = params->train_params.scene_name[0];
    std::shared_ptr<SceneBase> scene = std::make_shared<SceneBase>(params->train_params.scene_dir + "/" + scene_name);
    scene->train_indices =
        TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/train.txt");
    scene->test_indices =
        TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/eval.txt");
    scene->params = params;
    scene->LoadImagesCT(scene->train_indices);
    scene->LoadImagesCT(scene->test_indices);
    scene->Finalize();

    HyperTreeBase tree   = HyperTreeBase(3, params->octree_params.tree_depth);

    ts.neural_geometry = std::make_shared<GeometryExEx>(scene->num_channels, scene->D, tree, params);
    ts.scene = scene;
    ts.tree  = tree;
    // ts.LoadCheckpoint(params->train_params.checkpoint_directory);
    // ts.tree -> to(device);
    // ts.neural_geometry->to(device);
    return ts;
}

// double EvalStepProjection(TrainScene& ts, std::vector<int> indices, std::string name, int epoch_id,
//                               std::string checkpoint_name, bool test);

int main()
{
    // TrainScene ts;
    // ts.LoadCheckpoint("/storage/NeAT/Experiments/p2_15@2023-07-20_15-00-16/ep0009");

    std::string exp_dir = "/storage/NeAT/Experiments/p2_10@2023-07-28_15-12-41";
    std::string checkpoint_name = "ep0002";

    CombinedParams params;
    params.Load(exp_dir + "/params.ini");
    std::cout << "Loaded params from " << exp_dir + "/params.ini" << std::endl;
    std::cout << "Scene dir: " << params.train_params.scene_dir << std::endl;
    std::cout << "Scene name: " << params.train_params.scene_name << std::endl;

    TrainScene ts = TrainScene();
    initializeScene(ts, std::make_shared<CombinedParams>(params));
    std::cout << ts.scene->pose << std::endl;

    std::cout << ts.scene -> frames.size() << std::endl;
    ts.LoadCheckpoint(exp_dir + "/" + checkpoint_name);

    return 0;
}

// #include // 