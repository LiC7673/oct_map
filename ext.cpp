#include <torch/extension.h>
#include "tree.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_root", &launch_init_root, "Initialize Octree Root Node");
    m.def("update_octree", &launch_update_octree, "Update Octree from Depth Image");
    m.def("find_best_per_grid", &launch_find_best_per_grid, "Find best voxel per leaf grid");
}