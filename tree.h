#ifndef OCT_TREE_H_INCLUDED
#define OCT_TREE_H_INCLUDED
#include <torch/extension.h>
#include <vector_functions.h>
#include <cmath>
#include"config.h"

// 使用 cooperative_groups 命名空间



void launch_init_root(
    torch::Tensor d_node_pool,
    torch::Tensor d_allocators,
    torch::Tensor scene_bbox // (2, 3) [min, max]
);

void launch_find_best_per_grid(
    torch::Tensor d_leaf_pool,
    torch::Tensor d_node_pool,
    uint32_t      num_leaf_grids,
    float         voxel_size,
    float         weight_uncertainty,
    float         weight_frontier,
    torch::Tensor d_intermediate_buffer // 输出
);
/**
 * @brief (C++ Host) 启动主更新内核
 */
void launch_update_octree(
    // 输入
    torch::Tensor depth_map,     // (H, W)
    torch::Tensor intrinsics_t,  // (4,) fx, fy, cx, cy
    torch::Tensor pose_t,        // (4, 4)
    uint32_t      frame_id,
    float         max_dist,

    // 树 (I/O)
    torch::Tensor d_node_pool,
    torch::Tensor d_leaf_pool,
    torch::Tensor d_allocators,

    // 配置
    float         voxel_size,
    float         hit_weight,
    float         free_weight
);
#endif
// --- 绑定到 Python 的胶水代码 ---
