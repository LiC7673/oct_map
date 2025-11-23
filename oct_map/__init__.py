import torch
import numpy as np
from  sklearn.cluster import KMeans
# 1. 导入你刚刚编译的 C++/CUDA 模块
try:
    import oct_map._C
except ImportError:
    print("CUDA backend not built. Please run 'pip install .' in the setup.py directory.")
    exit(1)


class ActiveOctreeMap:
    # 2. 从 CUDA 代码中获取结构体的大小 (硬编码)
    # InternalNode: 8*uint32 + float4 = 8*4 + 4*4 = 32 + 16 = 48 字节
    NODE_STRUCT_SIZE_BYTES = 48

    # VoxelData: 6*float + float3 + uint32 = 6*4 + 12 + 4 = 24 + 12 + 4 = 40 字节
    # VoxelData 是 32 字节
    VOXEL_STRUCT_SIZE_BYTES = 32

    # 叶节点块的维度 (在 .cu 中硬编码为 8)
    LEAF_DIM = 8
    LEAF_GRID_SIZE = LEAF_DIM * LEAF_DIM * LEAF_DIM  # 512
    LEAF_GRID_SIZE_BYTES = LEAF_GRID_SIZE * VOXEL_STRUCT_SIZE_BYTES  # 512 * 32 = 16384 字节

    def __init__(self,
                 scene_bbox,  # 场景边界: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
                 voxel_size,  # 叶节点体素的真实大小 (e.g., 0.1m)

                 # --- 内存池配置 ---
                 initial_nodes=10000,
                 initial_leaves=10000,

                 # --- 更新权重 (来自 .cu) ---
                 hit_weight=0.85,
                 free_weight=-0.4,
                 max_ray_dist=5.0,
                 device='cuda'):

        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError("ActiveOctreeMap 必须在 CUDA 设备上运行")

        # --- 1. 存储配置 ---
        self.bbox_tensor = torch.tensor(scene_bbox, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.leaf_block_voxel_size = self.voxel_size * self.LEAF_DIM  # e.g., 0.8m
        self.hit_weight = hit_weight
        self.free_weight = free_weight
        self.max_ray_dist = max_ray_dist
        self.frame_id = 0
        print("init_fig")
        # --- 2. 分配 GPU 内存池 ---
        # 这些张量就是你的"树"的状态

        # 节点池: 存储 InternalNode (48 字节)
        # 我们使用 torch.uint8 作为原始字节缓冲区
        self.node_pool = torch.zeros(initial_nodes * self.NODE_STRUCT_SIZE_BYTES,
                                     dtype=torch.uint8, device=self.device).cuda()

        # 叶子池: 存储 VoxelData 块 (16384 字节)
        self.leaf_pool = torch.zeros(initial_leaves * self.LEAF_GRID_SIZE_BYTES,
                                     dtype=torch.uint8, device=self.device).cuda()

        # 分配器: [node_allocator, leaf_allocator]
        self.allocators = torch.zeros(2, dtype=torch.int32, device=self.device).cuda()

        print("init_node_pool")
        # --- 3. 调用 CUDA 内核初始化根节点 ---
        oct_map._C.init_root(
            self.node_pool,  # 传入内存池
            self.allocators,  # 传入分配器
            self.bbox_tensor  # 传入场景边界
        )
        print("init_node_pool")
        print(f"Octree 初始化完毕。")
        print(f"  节点池: {initial_nodes} 个节点, "
              f"{(self.node_pool.numel() * self.node_pool.element_size() / 1024 ** 2):.2f} MB")
        print(f"  叶子池: {initial_leaves} 个块, {(self.leaf_pool.numel() * self.leaf_pool.element_size() / 1024 ** 2):.2f}  MB")

        # print(f"  节点池: {initial_nodes} 个节点, "
        #       f"{(self.node_pool.numel() * self.node_pool.element_size() / 1024 ** 2):.2f} MB")
    def _check_and_resize_pools(self):
        """
        (关键) 检查内存池是否即将耗尽，如果需要则进行扩容。
        这是一个简化的实现；生产环境中需要更鲁棒的检查。
        """
        # 从 GPU 获取当前的分配计数
        node_count = self.allocators[0].item()
        leaf_count = self.allocators[1].item()

        node_capacity = self.node_pool.shape[0] // self.NODE_STRUCT_SIZE_BYTES
        leaf_capacity = self.leaf_pool.shape[0] // self.LEAF_GRID_SIZE_BYTES

        # 检查节点池
        if node_count >= node_capacity * 0.9:
            new_capacity = int(node_capacity * 1.5)
            print(f"WARNING: 节点池扩容 {node_capacity} -> {new_capacity}")
            new_pool = torch.zeros(new_capacity * self.NODE_STRUCT_SIZE_BYTES,
                                   dtype=torch.uint8, device=self.device)
            new_pool[:self.node_pool.shape[0]] = self.node_pool  # 复制旧数据
            self.node_pool = new_pool

        # 检查叶子池
        if leaf_count >= leaf_capacity * 0.9:
            new_capacity = int(leaf_capacity * 1.5)
            print(f"WARNING: 叶子池扩容 {leaf_capacity} -> {new_capacity}")
            new_pool = torch.zeros(new_capacity * self.LEAF_GRID_SIZE_BYTES,
                                   dtype=torch.uint8, device=self.device)
            new_pool[:self.leaf_pool.shape[0]] = self.leaf_pool  # 复制旧数据
            self.leaf_pool = new_pool

    def update(self, depth_map, pose, intrinsics):
        """
        使用单张深度图和相机位姿更新八叉树。

        :param depth_map: (H, W) torch.Tensor, float32,
        :param pose: (4, 4) torch.Tensor, float32, (相机到世界)
        :param intrinsics: (4,) torch.Tensor, float32, [fx, fy, cx, cy]
        """

        # 1. 确保所有输入都在 CUDA 上
        depth_map = depth_map.to(self.device, non_blocking=True)
        pose = pose.to(self.device, non_blocking=True)
        intrinsics_t = intrinsics.to(self.device, non_blocking=True)

        # 2. (可选但推荐) 在启动内核前检查并扩容
        self._check_and_resize_pools()

        # 3. 调用 CUDA 更新内核
        oct_map._C.update_octree(
            # --- 输入 ---
            depth_map,
            intrinsics_t,
            pose,
            self.frame_id,
            self.max_ray_dist,

            # --- 树的状态 (I/O) ---
            self.node_pool,
            self.leaf_pool,
            self.allocators,

            # --- 配置 ---
            self.voxel_size,
            self.hit_weight,
            self.free_weight
        )

        # 4. 更新帧计数器
        self.frame_id += 1

        # (此时，self.node_pool 和 self.leaf_pool 已被 CUDA 就地更新)

    def query_candidate_views(self, num_candidates=10, view_distance=1.5):
        """
        生成候选视角的主入口。

        参数:
            num_candidates (int): 需要生成的视角数量。
            view_distance (float): 相机距离目标表面的理想观测距离（米）。

        返回:
            torch.Tensor: (N, 4, 4) 候选位姿矩阵 (World-to-Camera 或 Camera-to-World，取决于你的约定)
                          这里返回的是 Camera-to-World (T_cw)，即相机在世界坐标系下的位姿。
        """

        # 1. 获取原始目标数据 [score, x, y, z, nx, ny, nz, timestamp]
        # (调用上一节写的 CUDA 接口)
        raw_targets = self._fetch_raw_targets_from_cuda()

        if raw_targets is None or raw_targets.shape[0] == 0:
            print("[Query] 未找到高信息增益区域。")
            return torch.empty((0, 4, 4), device=self.device)

        # 2. 数据预处理 (转到 CPU 进行聚类)
        # targets: (K, 6) -> [x, y, z, nx, ny, nz]
        # 我们只关心位置和法线用于聚类
        targets_np = raw_targets[:, 1:7].cpu().numpy()
        scores_np = raw_targets[:, 0].cpu().numpy()

        # 过滤掉非法法线 (长度为0的)
        norms = np.linalg.norm(targets_np[:, 3:6], axis=1)
        valid_mask = norms > 0.1
        targets_np = targets_np[valid_mask]
        scores_np = scores_np[valid_mask]

        if len(targets_np) == 0:
            return torch.empty((0, 4, 4), device=self.device)

        # 归一化法线 (聚类对尺度敏感)
        # 位置和法线的量纲不同，通常需要加权。
        # 简单的做法是直接聚类，或者给位置乘一个权重。
        # 这里我们直接对 6D 向量 [x,y,z, nx,ny,nz] 进行聚类

        # 3. 执行 K-Means 聚类
        # 实际产生的聚类数可能少于 num_candidates (如果点太少)
        n_clusters = min(num_candidates, len(targets_np))
        kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)

        # 使用分数作为样本权重！
        # 这样聚类中心会偏向分数高的地方
        kmeans.fit(targets_np, sample_weight=scores_np)

        cluster_centers = kmeans.cluster_centers_  # (N, 6)

        # 4. 将聚类中心转换为相机位姿
        candidate_poses = []

        for center in cluster_centers:
            target_pos = torch.tensor(center[0:3], device=self.device, dtype=torch.float32)
            target_normal = torch.tensor(center[3:6], device=self.device, dtype=torch.float32)

            # 归一化法线
            target_normal = torch.nn.functional.normalize(target_normal, dim=0)

            # 生成位姿
            pose = self._compute_lookat_pose(target_pos, target_normal, view_distance)
            candidate_poses.append(pose)

        return torch.stack(candidate_poses)

    def _fetch_raw_targets_from_cuda(self):
        """
        辅助函数：调用 CUDA 内核并整理输出。
        """
        current_leaf_count = self.allocators[1].item()
        if current_leaf_count == 0: return None

        # 分配缓冲区
        intermediate = torch.full((current_leaf_count, 8), -1.0, device=self.device)

        # 调用 CUDA (假设你已经绑定好了)

        oct_map._C.find_best_per_grid(
            self.leaf_pool,
            self.node_pool,
            current_leaf_count,
            self.voxel_size,
            1.0, 0.5,  # 权重
            intermediate
        )

        # 筛选有效行 (score > 0)
        mask = intermediate[:, 0] > 0
        valid_data = intermediate[mask]

        # --- 关键修正：填补缺失的 x,y,z ---
        # 由于之前的 Kernel 没算 xyz，这里做一个简易补救：
        # 我们无法精确恢复 xyz，除非我们在 VoxelData 里存了，或者有一个并行数组存了 LeafOrigin。
        #
        # **强烈建议**：在 C++ Host 端维护一个 leaf_origins 数组。
        #
        # 临时方案：假设 valid_data 里的 xyz 已经是有效的（如果你按我上一条建议修改了Kernel），
        # 或者我们在 VoxelData 里加了 world_pos 字段。

        return valid_data

    def _compute_lookat_pose(self, target_pos, target_normal, distance):
        """
        计算 LookAt 矩阵。

        策略：
        1. 相机位置 = 目标点 + 法线 * 距离
           (假设法线指向表面外侧，我们想看表面，所以相机要在法线方向上)
        2. 相机 -Z 轴 = 指向目标 (Target - Eye)
           (这是 OpenGL/NeRF 的标准约定，相机看向 -Z)
        """
        # 1. 计算相机中心位置 (Eye)
        # 我们沿着法线方向往外退 distance 米
        eye_pos = target_pos + target_normal * distance

        # 2. 计算基向量
        # Forward (Z-axis): 应该指向相机后方 (OpenGL约定)
        # 我们希望相机看向 -Z，也就是看向 target_pos
        # 所以 +Z 应该指向 target_pos 的反方向，也就是 target_normal
        forward = target_normal

        # World Up (Y-axis)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)

        # 处理万向节死锁 (如果 forward 和 world_up 平行)
        if torch.abs(torch.dot(forward, world_up)) > 0.99:
            world_up = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        # Right (X-axis) = Up cross Forward
        right = torch.cross(world_up, forward)
        right = torch.nn.functional.normalize(right, dim=0)

        # True Up (Y-axis) = Forward cross Right
        up = torch.cross(forward, right)
        up = torch.nn.functional.normalize(up, dim=0)

        # 3. 构建 4x4 矩阵 (Camera-to-World)
        # R = [right, up, forward] (列向量)
        pose = torch.eye(4, device=self.device)
        pose[0:3, 0] = right
        pose[0:3, 1] = up
        pose[0:3, 2] = forward
        pose[0:3, 3] = eye_pos

        return pose