#ifndef OCT_UTILS_H_INCLUDED
#define OCT_UTILS_H_INCLUDED
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h> // 用于初始化新叶子块
#include <cuda_runtime.h> // 确保你包含了 CUDA 头文件，float3 在这里定义
#include <cub/cub.cuh>


typedef cub::KeyValuePair<float, int> FloatIntPair;
__device__ __forceinline__ bool operator>(const FloatIntPair& a, const FloatIntPair& b)
{
    // 我们定义“大于” = “key (浮点数分数) 更高”
    return a.key > b.key;
}
// 加法: a + b
__host__ __device__ __forceinline__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// 减法: a - b
__host__ __device__ __forceinline__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// 乘法 (分量相乘): a * b
__host__ __device__ __forceinline__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// 除法 (分量相除): a / b
__host__ __device__ __forceinline__ float3 operator/(const float3& a, const float3& b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

// --- float3 与标量 (float) 之间的运算 ---

// 标量乘法: v * s
__host__ __device__ __forceinline__ float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// 标量乘法 (交换律): s * v
__host__ __device__ __forceinline__ float3 operator*(float s, const float3& a)
{
    return make_float3(s * a.x, s * a.y, s * a.z);
}

// 标量除法: v / s
__host__ __device__ __forceinline__ float3 operator/(const float3& a, float s)
{
    // 为标量除法做一个小优化，使用倒数乘法
    float inv_s = 1.0f / s;
    return make_float3(a.x * inv_s, a.y * inv_s, a.z * inv_s);
}

// --- 一元运算符 ---

// 负号: -v
__host__ __device__ __forceinline__ float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// --- 就地（In-place）运算符 ---

// 就地加法: a += b
__host__ __device__ __forceinline__ void operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

// 就地减法: a -= b
__host__ __device__ __forceinline__ void operator-=(float3& a, const float3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

// 就地标量乘法: a *= s
__host__ __device__ __forceinline__ void operator*=(float3& a, float s)
{
    a.x *= s;
    a.y *= s;
    a.z *= s;
}

// 就地标量除法: a /= s
__host__ __device__ __forceinline__ void operator/=(float3& a, float s)
{
    float inv_s = 1.0f / s;
    a.x *= inv_s;
    a.y *= inv_s;

    a.z *= inv_s;
}

__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

/**
 * @brief (设备端) 归一化 float3 向量
 */
__device__ __forceinline__ float3 normalize(const float3& v) {
    float invLen = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

// --- 两个 float3 之间的运算 ---
// typedef cub::KeyValuePair<float, int> FloatIntPair;
// __device__ __forceinline__ bool operator>(const FloatIntPair& a, const FloatIntPair& b);
//
// __host__ __device__ __forceinline__ float3 operator+(const float3& a, const float3& b);
//
//
// __host__ __device__ __forceinline__ float3 operator-(const float3& a, const float3& b);
//
//
// __host__ __device__ __forceinline__ float3 operator*(const float3& a, const float3& b);
//
//
// __host__ __device__ __forceinline__ float3 operator/(const float3& a, const float3& b);
//
// // --- float3 与标量 (float) 之间的运算 ---
//
// // 标量乘法: v * s
// __host__ __device__ __forceinline__ float3 operator*(const float3& a, float s);
//
// // 标量乘法 (交换律): s * v
// __host__ __device__ __forceinline__ float3 operator*(float s, const float3& a);
//
// // 标量除法: v / s
// __host__ __device__ __forceinline__ float3 operator/(const float3& a, float s);
//
// // --- 一元运算符 ---
//
// // 负号: -v
// __host__ __device__ __forceinline__ float3 operator-(const float3& a);
//
// // --- 就地（In-place）运算符 ---
//
// // 就地加法: a += b
// __host__ __device__ __forceinline__ void operator+=(float3& a, const float3& b);
// // 就地减法: a -= b
// __host__ __device__ __forceinline__ void operator-=(float3& a, const float3& b);
//
// // 就地标量乘法: a *= s
// __host__ __device__ __forceinline__ void operator*=(float3& a, float s);
//
// // 就地标量除法: a /= s
// __host__ __device__ __forceinline__ void operator/=(float3& a, float s);
//
// __device__ __forceinline__ float length(const float3& v);
//
// /**
//  * @brief (设备端) 归一化 float3 向量
//  */
// __device__ __forceinline__ float3 normalize(const float3& v);
#endif