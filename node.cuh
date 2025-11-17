
#ifndef OCT_NODE_H_INCLUDED
#define OCT_NODE_H_INCLUDED
// --- 指针编码的常量 ---

#include"config.h"// 1. 设备端数据结构 (来自我们之前的设计)
// ------------------------------------------------------------------

// --- 指针编码的常量 ---
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h> // 用于初始化新叶子块
#include <stdio.h>
#include"node.h"
// 使用 cooperative_groups 命名空间
namespace cg = cooperative_groups;

/**
 * @struct VoxelData (32 字节, 零填充)
 */
struct VoxelData {
    float log_odds;
    float sh_l0_weight;
    float3 sh_l1_vec;
    float nbv_uncertainty_score;
    float nbv_frontier_score;
    uint32_t last_updated_timestamp;

    __device__ VoxelData() :
        log_odds(0.0f),
        sh_l0_weight(0.0f),
        sh_l1_vec(make_float3(0.0f, 0.0f, 0.0f)),
        nbv_uncertainty_score(0.0f),
        nbv_frontier_score(0.0f),
        last_updated_timestamp(0)
    {}

    /**
     * @brief (设备端) 更新此体素的观测信息
     */
    __device__ void update_view_sh(const float3& view_dir, float weight) {
        atomicAdd(&this->sh_l0_weight, weight);
        atomicAdd(&this->sh_l1_vec.x, view_dir.x * weight);
        atomicAdd(&this->sh_l1_vec.y, view_dir.y * weight);
        atomicAdd(&this->sh_l1_vec.z, view_dir.z * weight);
    }
};

/**
 * @struct InternalNode (48 字节)
 */
struct InternalNode {
    uint32_t children[8];
    float4 center_and_half_size; // .xyz = center, .w = half_size

    __device__ InternalNode() {
        for (int i = 0; i < 8; ++i) children[i] = NULL_PTR;
        center_and_half_size = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    /**
     * @brief (设备端) 使用父节点信息进行初始化
     */
    __device__ void init_from_parent(const float4& parent_c_hs, int child_idx) {
        float child_half_size = parent_c_hs.w * 0.5f;
        float child_quarter_size = child_half_size * 0.5f;

        center_and_half_size = make_float4(
            parent_c_hs.x + ((child_idx & 1) ? child_quarter_size : -child_quarter_size),
            parent_c_hs.y + ((child_idx & 2) ? child_quarter_size : -child_quarter_size),
            parent_c_hs.z + ((child_idx & 4) ? child_quarter_size : -child_quarter_size),
            child_half_size
        );
        for (int i = 0; i < 8; ++i) children[i] = NULL_PTR;
    }
};

#endif