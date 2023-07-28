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
#include "utils/utils.h"

#include "build_config.h"
#include "geometry/geometry_ex_ex.h"
#include "tensorboard_logger.h"
#include "utils/cimg_wrapper.h"
using namespace Saiga;

int main()
{
    std::cout << "Hello World!" << std::endl;
    return 0;
}

// #include 

// struct TrainScene
// {
//     std::shared_ptr<SceneBase> scene;
//     HyperTreeBase tree = nullptr;
//     std::shared_ptr<HierarchicalNeuralGeometry> neural_geometry;

//     double last_eval_loss           = 9237643867809436;
//     double new_eval_loss            = 9237643867809436;
//     int last_structure_change_epoch = 0;

//     void SaveCheckpoint(const std::string& dir)
//     {
//         auto prefix = dir + "/" + scene->scene_name + "_";
//         scene->SaveCheckpoint(dir);

//         torch::nn::ModuleHolder<HierarchicalNeuralGeometry> holder(neural_geometry);
//         torch::save(holder, prefix + "geometry.pth");

//         torch::save(tree, prefix + "tree.pth");
//     }

//     void LoadCheckpoint(const std::string& dir)
//     {
//         auto prefix = dir + "/" + scene->scene_name + "_";
//         scene->LoadCheckpoint(dir);

//         if (std::filesystem::exists(prefix + "geometry.pth"))
//         {
//             std::cout << "Load checkpoint geometry " << std::endl;
//             torch::nn::ModuleHolder<HierarchicalNeuralGeometry> holder(neural_geometry);
//             torch::load(holder, prefix + "geometry.pth");
//         }
//         if (std::filesystem::exists(prefix + "tree.pth"))
//         {
//             torch::load(tree, prefix + "tree.pth");
//         }
//     }
// };