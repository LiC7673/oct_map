#include"tree.h"
#include"utils.cuh"
#include"node.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h> // 用于初始化新叶子块
#include <stdio.h>
#include <cub/cub.cuh>
namespace cg = cooperative_groups;
__device__ __forceinline__ bool is_leaf_ptr(uint32_t ptr) {
    return (ptr & LEAF_FLAG) != 0;
}

__device__ __forceinline__ uint32_t get_ptr_index(uint32_t ptr) {
    return (ptr & INDEX_MASK);
}

__device__ __forceinline__ int get_child_index_from_pos(const float3& p, const float3& node_center) {
    int idx = 0;
    if (p.x > node_center.x) idx |= 1; // +X
    if (p.y > node_center.y) idx |= 2; // +Y
    if (p.z > node_center.z) idx |= 4; // +Z
    return idx;
}//根据xxx定位树八个所在出
__device__ __forceinline__ float4 get_child_bbox(
    const float4& parent_c_hs,
    int child_idx
) {
    float child_half_size = parent_c_hs.w * 0.5f;
    float child_quarter_size = child_half_size * 0.5f;

    return make_float4(
        parent_c_hs.x + ((child_idx & 1) ? child_quarter_size : -child_quarter_size),
        parent_c_hs.y + ((child_idx & 2) ? child_quarter_size : -child_quarter_size),
        parent_c_hs.z + ((child_idx & 4) ? child_quarter_size : -child_quarter_size),
        child_half_size
    );
}
/**
 * @brief (设备端) 初始化一个新分配的叶子块 (使用一个完整的 CUDA Block)
 */
__global__ void init_leaf_grid_kernel(VoxelData* leaf_grid_start) {
    // 使用一个 512 线程的 block
    int i = threadIdx.x;
    if (i < LEAF_GRID_SIZE) {
        // 调用默认构造函数
        new (&leaf_grid_start[i]) VoxelData();
    }
}

/**
 * @brief (设备端) 这是核心遍历与分配函数
 * @return 指向目标 VoxelData 的指针，如果失败则为 nullptr
 */
__device__ __forceinline__ VoxelData* find_or_create_voxel_at(
    const float3&  world_pos,           // 目标世界坐标
    InternalNode* d_node_pool,         // [IO] 节点内存池
    VoxelData* d_leaf_pool,         // [IO] 叶子内存池
    uint32_t* d_node_allocator,    // [IO] 节点分配器
    uint32_t* d_leaf_allocator,    // [IO] 叶子分配器
    const float    LEAF_BLOCK_H_SIZE    // 叶子块的半边长
) {
    // 1. 从根节点 (index=0) 开始
    uint32_t node_idx = 0;
    InternalNode* node = &d_node_pool[0];
    float4 current_c_hs = node->center_and_half_size;

    for (int depth = 0; depth < 20; ++depth) { // 限制最大深度

        // 2. 确定子节点
        int child_idx_in_node = get_child_index_from_pos(world_pos, make_float3(current_c_hs.x, current_c_hs.y, current_c_hs.z));
        uint32_t child_ptr = node->children[child_idx_in_node];

        // 3. 如果子节点不存在，创建它 (最关键的部分)
        if (child_ptr == NULL_PTR) {

            // 检查我们是否在叶子层级
            bool is_leaf_level = (current_c_hs.w * 0.5f) <= LEAF_BLOCK_H_SIZE;
            uint32_t new_ptr_val = NULL_PTR;

            if (is_leaf_level) {
                // --- 3a. 分配一个新的叶子块 ---
                uint32_t new_leaf_idx = atomicAdd(d_leaf_allocator, 1);
                new_ptr_val = LEAF_FLAG | new_leaf_idx; // MSB=1 | index

                // ** 关键 **: 启动一个内核来初始化这个新块
                // (这需要 CUDA 动态并行，或更简单的：
                // 假设内存在主机端 resize 时已被清零)
                // 为简单起见，我们假设主机端会处理清零。

            } else {
                // --- 3b. 分配一个新的内部节点 ---
                uint32_t new_node_idx = atomicAdd(d_node_allocator, 1);
                new_ptr_val = new_node_idx; // MSB=0 | index
            }

            // 4. 尝试用 CAS (Compare-and-Swap) 写入新指针
            uint32_t old_val = atomicCAS(&node->children[child_idx_in_node], NULL_PTR, new_ptr_val);

            if (old_val == NULL_PTR) {

                child_ptr = new_ptr_val;

                if (!is_leaf_level) {
                    // 我们创建了一个 *内部节点*，需要初始化它
                    InternalNode* new_node = &d_node_pool[get_ptr_index(child_ptr)];
                    new_node->init_from_parent(current_c_hs, child_idx_in_node);
                }
                // (如果是叶子块，我们假设它已在主机端被清零)

            } else {
                // --- 4b. 我们输了！另一个线程在我们之前写入了 ---
                child_ptr = old_val; // 使用另一个线程创建的指针
                // (我们刚分配的 idx 变成了孤儿，没关系，这是标准做法)
            }
        }

        // 5. 解码指针
        if (is_leaf_ptr(child_ptr)) {
            // 6. 到达叶子块！
            uint32_t leaf_idx = get_ptr_index(child_ptr);
            VoxelData* leaf_grid_start = &d_leaf_pool[leaf_idx * LEAF_GRID_SIZE];

            // 获取子节点的bbox (我们必须从父节点计算它)
            float4 leaf_c_hs = node->center_and_half_size; // 父节点的
            leaf_c_hs = get_child_bbox(leaf_c_hs, child_idx_in_node);

            // 计算在 8x8x8 块中的局部索引 (x,y,z)
            float voxel_size = leaf_c_hs.w * 2.0f / LEAF_DIM;
            float3 leaf_origin = make_float3(leaf_c_hs.x, leaf_c_hs.y, leaf_c_hs.z);
            float3 leaf_min_corner = make_float3(
                leaf_origin.x - leaf_c_hs.w,
                leaf_origin.y - leaf_c_hs.w,
                leaf_origin.z - leaf_c_hs.w
            );

            int local_x = (int)((world_pos.x - leaf_min_corner.x) / voxel_size);
            int local_y = (int)((world_pos.y - leaf_min_corner.y) / voxel_size);
            int local_z = (int)((world_pos.z - leaf_min_corner.z) / voxel_size);

            local_x = max(0, min(LEAF_DIM - 1, local_x));
            local_y = max(0, min(LEAF_DIM - 1, local_y));
            local_z = max(0, min(LEAF_DIM - 1, local_z));

            int voxel_idx_in_leaf = (local_z * LEAF_DIM * LEAF_DIM) + (local_y * LEAF_DIM) + local_x;

            return &leaf_grid_start[voxel_idx_in_leaf];

        } else {
            // 7. 继续深入八叉树
            node_idx = get_ptr_index(child_ptr);
            node = &d_node_pool[node_idx];
            current_c_hs = node->center_and_half_size;
        }
    }

    return nullptr; // 超过最大深度，失败
}


// ------------------------------------------------------------------
// 3. 主更新内核 (一个线程一条光线)
// ------------------------------------------------------------------
__global__ void update_octree_kernel(
    // --- 输入数据 ---
    const float* d_depth_map,     // H x W
    const float4   intrinsics,      // fx, fy, cx, cy
    const float* d_pose,          // 4x4 (row-major)
    int            img_width,
    int            img_height,
    float          max_ray_dist,
    uint32_t       current_frame_id,

    // --- 树数据结构 ---
    InternalNode* d_node_pool,
    VoxelData* d_leaf_pool,
    uint32_t* d_node_allocator,
    uint32_t* d_leaf_allocator,
    const float    LEAF_BLOCK_V_SIZE, // 体素的真实大小 (e.g., 0.1m)

    // --- 更新权重 ---
    const float    HIT_WEIGHT,
    const float    FREE_WEIGHT
) {
    // --- 1. 计算线程ID和像素坐标 ---
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= img_width || v >= img_height) {
        return;
    }
    int pixel_idx = v * img_width + u;

    // --- 2. 从输入计算光线 (Origin, Dir, HitDist) ---
    float depth = d_depth_map[pixel_idx];
    bool is_hit = (depth > 0.01f && depth < max_ray_dist);
    float hit_dist = is_hit ? depth : max_ray_dist;

    // 像素坐标 -> 相机坐标
    float3 cam_pos = make_float3(
        (u - intrinsics.z) / intrinsics.x * depth,
        (v - intrinsics.w) / intrinsics.y * depth,
        depth
    );

    // 相机坐标 -> 世界坐标
    float3 ray_origin = make_float3(d_pose[3], d_pose[7], d_pose[11]);
    float3 world_pos_hit = make_float3(
        d_pose[0] * cam_pos.x + d_pose[1] * cam_pos.y + d_pose[2] * cam_pos.z + d_pose[3],
        d_pose[4] * cam_pos.x + d_pose[5] * cam_pos.y + d_pose[6] * cam_pos.z + d_pose[7],
        d_pose[8] * cam_pos.x + d_pose[9] * cam_pos.y + d_pose[10] * cam_pos.z + d_pose[11]
    );

    float3 ray_dir = world_pos_hit - ray_origin;
    // (注意: ray_dir 的长度是 'depth', 而不是 1.0)
    ray_dir = normalize(ray_dir);

    float3 view_dir = -ray_dir; // 视图方向
    float LEAF_BLOCK_H_SIZE = (LEAF_BLOCK_V_SIZE * LEAF_DIM) * 0.5f;

    // --- 3. 光线行进 (Ray Marching) - 更新 "Free" 空间 ---
    // (这是一个简化的步进。一个优化的版本会使用Slab-Test)
    for (float t = 0.0f; t < hit_dist - LEAF_BLOCK_V_SIZE; t += LEAF_BLOCK_V_SIZE) {
        float3 current_pos = ray_origin + ray_dir * t;

        VoxelData* voxel = find_or_create_voxel_at(
            current_pos, d_node_pool, d_leaf_pool, d_node_allocator, d_leaf_allocator,
            LEAF_BLOCK_H_SIZE
        );

        if (voxel != nullptr) {
            atomicAdd(&voxel->log_odds, FREE_WEIGHT);
            voxel->update_view_sh(view_dir, 0.5f); // 用较低的权重更新 free 空间
            atomicExch(&voxel->last_updated_timestamp, current_frame_id);
            // (可以在这里更新 Frontier Score)
        }
    }

    // --- 4. 更新 "Hit" 体素 (如果光线击中了) ---
    if (is_hit) {
        float3 hit_pos = ray_origin + ray_dir * hit_dist;
        VoxelData* hit_voxel = find_or_create_voxel_at(
            hit_pos, d_node_pool, d_leaf_pool, d_node_allocator, d_leaf_allocator,
            LEAF_BLOCK_H_SIZE
        );

        if (hit_voxel != nullptr) {
            atomicAdd(&hit_voxel->log_odds, HIT_WEIGHT);
            hit_voxel->update_view_sh(view_dir, 1.0f); // 用 1.0 的权重更新 hit 空间
            atomicExch(&hit_voxel->last_updated_timestamp, current_frame_id);

            // 更新 NBV 不确定性分数
            float unc_score = 1.0f - tanhf(fabsf(hit_voxel->log_odds));
            float view_score = 1.0f - (length(hit_voxel->sh_l1_vec) / (hit_voxel->sh_l0_weight + 1e-6f));
            atomicExch(&hit_voxel->nbv_uncertainty_score, max(unc_score, view_score));
        }
    }
}


// ------------------------------------------------------------------
// 6. 查询内核 (查找每个网格中的最佳体素)
// ------------------------------------------------------------------

// CUB 需要一个 Key-Value 对来查找最大值


__global__ void find_best_voxel_per_grid_kernel(
    // --- 输入 ---
    const VoxelData* d_leaf_pool,
    const InternalNode* d_node_pool,
    uint32_t         num_leaf_grids,
    float            voxel_size,
    float            weight_uncertainty, // 贪心算法的权重
    float            weight_frontier,    // 贪心算法的权重

    // --- 输出 (中间缓冲区) ---
    // [score, x, y, z, nx, ny, nz, timestamp]
    float* d_intermediate_buffer
) {
    // --- CUB 减少所需的共享内存 ---
    __shared__ cub::BlockReduce<FloatIntPair, LEAF_GRID_SIZE>::TempStorage temp_storage;

    // --- 1. 识别我们的 Block 和 Thread ---

    uint32_t grid_idx = blockIdx.x;
    if (grid_idx >= num_leaf_grids) {
        return;
    }

    // 每个 Thread 负责一个 Voxel
    int local_voxel_idx = threadIdx.x;


    const VoxelData* voxel = &d_leaf_pool[grid_idx * LEAF_GRID_SIZE + local_voxel_idx];

    // 贪心得分
    float score = (voxel->nbv_uncertainty_score * weight_uncertainty) +
                  (voxel->nbv_frontier_score * weight_frontier);

    // 封装成 CUB 的 Key-Value 对 (Key=score, Value=index)
    FloatIntPair my_voxel_score(score, local_voxel_idx);

    // --- 3. Reduce 阶段: 查找块内的最大值 ---
    // 所有 512 个线程协作，找到具有最大 'score' (Key) 的 'local_voxel_idx' (Value)
//     FloatIntPair best_voxel_in_grid = cub::BlockReduce<FloatIntPair, LEAF_GRID_SIZE>(temp_storage).Max(my_voxel_score);

    FloatIntPair best_voxel_in_grid = cub::BlockReduce<FloatIntPair, LEAF_GRID_SIZE>(temp_storage).Reduce(my_voxel_score, cub::Max());
    // --- 4. 线程 0 负责写回 ---
    if (threadIdx.x == 0) {
        int best_local_idx = best_voxel_in_grid.value;
        float best_score = best_voxel_in_grid.key;

        // 如果最佳分数 > 0, 我们才关心它
        if (best_score > 1e-3f) {
            const VoxelData* best_voxel = &d_leaf_pool[grid_idx * LEAF_GRID_SIZE + best_local_idx];

            // --- 5. 计算世界坐标和法线 ---
            // (这是最难的部分 - 我们需要反向遍历树来找到此网格的中心)
            // (简化: 假设我们无法轻松做到。一个更好的方法
            //  是让 update_kernel 也写入 VoxelData 的 world_pos)

            // 妥协：我们只返回法线和分数。
            // 我们将在Python中通过聚类来生成最终位置。
            float3 normal = make_float3(0.0f, 0.0f, 0.0f);
            if (best_voxel->sh_l0_weight > 1e-3f) {
                // L1 SH 球谐函数给我们平均方向
                normal = normalize(best_voxel->sh_l1_vec);
            }

            int offset = grid_idx * 8; // 我们的缓冲区有 8 个浮点数
            d_intermediate_buffer[offset + 0] = best_score;
            d_intermediate_buffer[offset + 1] = 0.0f; // TODO: X
            d_intermediate_buffer[offset + 2] = 0.0f; // TODO: Y
            d_intermediate_buffer[offset + 3] = 0.0f; // TODO: Z
            d_intermediate_buffer[offset + 4] = normal.x;
            d_intermediate_buffer[offset + 5] = normal.y;
            d_intermediate_buffer[offset + 6] = normal.z;
            d_intermediate_buffer[offset + 7] = (float)best_voxel->last_updated_timestamp;
        } else {
             d_intermediate_buffer[grid_idx * 8 + 0] = -1.0f; // 标记为无效
        }
    }
}


// ------------------------------------------------------------------
// 7. C++ 绑定 (添加新的)
// ------------------------------------------------------------------
// (在你的 PYBIND11_MODULE 宏里面)



// 对应的 (C++ Host) 启动器函数
void launch_find_best_per_grid(
    torch::Tensor d_leaf_pool,
    torch::Tensor d_node_pool,
    uint32_t      num_leaf_grids,
    float         voxel_size,
    float         weight_uncertainty,
    float         weight_frontier,
    torch::Tensor d_intermediate_buffer // 输出
) {
    TORCH_CHECK(d_leaf_pool.is_cuda(), "Leaf pool must be on CUDA");

    // Block 大小必须是 512 (LEAF_GRID_SIZE)
    const dim3 threadsPerBlock(LEAF_GRID_SIZE);
    const dim3 numBlocks(num_leaf_grids);

    find_best_voxel_per_grid_kernel<<<numBlocks, threadsPerBlock>>>(
        (const VoxelData*)d_leaf_pool.data_ptr(),
        (const InternalNode*)d_node_pool.data_ptr(),
        num_leaf_grids,
        voxel_size,
        weight_uncertainty,
        weight_frontier,
        d_intermediate_buffer.data_ptr<float>()
    );

}


__global__ void init_root_kernel(
    InternalNode* d_node_pool,
    uint32_t* d_allocators,
    float4 scene_center_and_half_size
) {
//     printf("init_root_kernel");
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 1. 初始化根节点 (index=0)
        new (&d_node_pool[0]) InternalNode();
        d_node_pool[0].center_and_half_size = scene_center_and_half_size;

        // 2. 初始化分配器
        d_allocators[0] = 1; // 节点分配器 (index 0 已被根节点占用)
        d_allocators[1] = 0; // 叶子分配器 (从 0 开始)
//         printf("finish_init");
    }
}
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
) {
    TORCH_CHECK(depth_map.is_cuda(), "Depth map must be on CUDA");
    printf("update_start");

    int img_height = depth_map.size(0);
    int img_width = depth_map.size(1);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (img_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (img_height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    float4 intrinsics;
    auto cpu_i =intrinsics_t.to(torch::kCPU).to(torch::kFloat32).contiguous();

    // 2. 对 CPU 上的副本创建访问器
//     auto bbox_acc = cpu_bbox.accessor<float, 2>();
    auto i_acc = cpu_i.accessor<float, 1>();
    intrinsics.x = i_acc[0]; intrinsics.y = i_acc[1];
    intrinsics.z = i_acc[2]; intrinsics.w = i_acc[3];
    printf("update_start");
    update_octree_kernel<<<numBlocks, threadsPerBlock>>>(
        depth_map.data_ptr<float>(),
        intrinsics,
        pose_t.data_ptr<float>(),
        img_width,
        img_height,
        max_dist,
        frame_id,
        (InternalNode*)d_node_pool.data_ptr(),
        (VoxelData*)d_leaf_pool.data_ptr(),
        (uint32_t*)d_allocators.data_ptr(),
        (uint32_t*)d_allocators.data_ptr() + 1, // leaf_allocator
        voxel_size,
        hit_weight,
        free_weight
    );
    // (不需要同步，让它异步运行)
}

void launch_init_root(
    torch::Tensor d_node_pool,//(N,nodes) N是总池子的大小
    torch::Tensor d_allocators,//()用于标记池子
    torch::Tensor scene_bbox // (2, 3) [min, max]
) {
//     printf("init_start");
    CHECK_CUDA(d_node_pool);
    CHECK_CUDA(d_allocators);
    CHECK_CUDA(scene_bbox);
    auto cpu_bbox = scene_bbox.to(torch::kCPU).to(torch::kFloat32).contiguous();

    // 2. 对 CPU 上的副本创建访问器
    auto bbox_acc = cpu_bbox.accessor<float, 2>();
    float4 c_hs;
//     auto bbox_acc = scene_bbox.accessor<float, 2>();
    c_hs.x = (bbox_acc[0][0] + bbox_acc[1][0]) * 0.5f; // center x
    c_hs.y = (bbox_acc[0][1] + bbox_acc[1][1]) * 0.5f; // center y
    c_hs.z = (bbox_acc[0][2] + bbox_acc[1][2]) * 0.5f; // center z
    c_hs.w = (bbox_acc[1][0] - bbox_acc[0][0]) * 0.5f; // half_size

//     printf("local_init");
    init_root_kernel<<<1, 1>>>(
        (InternalNode*)d_node_pool.data_ptr(),
        (uint32_t*)d_allocators.data_ptr(),
        c_hs
    );
    cudaDeviceSynchronize(); // 确保初始化完成
}
