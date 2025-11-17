import torch
import numpy as np

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
                 initial_leaves=1000,

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

        # --- 2. 分配 GPU 内存池 ---
        # 这些张量就是你的"树"的状态

        # 节点池: 存储 InternalNode (48 字节)
        # 我们使用 torch.uint8 作为原始字节缓冲区
        self.node_pool = torch.zeros(initial_nodes * self.NODE_STRUCT_SIZE_BYTES,
                                     dtype=torch.uint8, device=self.device)

        # 叶子池: 存储 VoxelData 块 (16384 字节)
        self.leaf_pool = torch.zeros(initial_leaves * self.LEAF_GRID_SIZE_BYTES,
                                     dtype=torch.uint8, device=self.device)

        # 分配器: [node_allocator, leaf_allocator]
        self.allocators = torch.zeros(2, dtype=torch.int32, device=self.device)

        # --- 3. 调用 CUDA 内核初始化根节点 ---
        oct_map._C.init_root(
            self.node_pool,  # 传入内存池
            self.allocators,  # 传入分配器
            self.bbox_tensor  # 传入场景边界
        )

        print(f"Octree 初始化完毕。")
        print(f"  节点池: {initial_nodes} 个节点, {(self.node_pool.nbytes / 1024 ** 2):.2f} MB")
        print(f"  叶子池: {initial_leaves} 个块, {(self.leaf_pool.nbytes / 1024 ** 2):.2f} MB")

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

    # --- TODO: 未来的查询函数 ---
    def query_candidate_views(self, num_candidates):
        """
        (这是下一步) 你需要编写另一个 CUDA 内核 (query_kernel)
        来并行读取 self.node_pool 和 self.leaf_pool，
        计算 NBV 分数，并返回候选视角。
        """
        print("查询功能尚未实现。")
        # 伪代码:
        # candidates = backend.query_views(
        #     self.node_pool,
        #     self.leaf_pool,
        #     num_candidates
        # )
        # return candidates
        pass