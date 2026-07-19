#pragma once 

#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>


constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;  // GQA: two Q heads share each KV head.
constexpr int HEAD_DIM = 128;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;   // 2048
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM; // 1024
constexpr int VOCAB_SIZE = 151936;
constexpr int NUM_LAYERS = 28;
constexpr float RMS_NORM_EPS = 1e-6f;

// EXECUTION CONSTANTS 
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE; // 8 warps per block.

// This checkpoint targets a full A100 with 108 SMs. A cooperative grid must be
// small enough for every block to be resident simultaneously.
constexpr int DECODE_NUM_BLOCKS = 108; 

constexpr int LM_HEAD_NUM_BLOCKS = 1184;
constexpr int LM_HEAD_BLOCK_SIZE = 256;

// REDUCTION HELPERS

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffff, value, offset)
        );
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(
    float value,
    float* shared_memory
) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    // Stage 1: reduce independently inside all eight warps. reduce within the lanes
    value = warp_reduce_sum(value);

    if (lane_id == 0) {
        shared_memory[warp_id] = value;
    }
    __syncthreads();

    // Stage 2: warp zero reduces the eight warp results.
    if (warp_id == 0) {
        value =
            (lane_id < NUM_WARPS)
            ? shared_memory[lane_id]
            : 0.0f;

        value = warp_reduce_sum(value);

        if (lane_id == 0) {
            shared_memory[0] = value;
        }
    }
    __syncthreads();

    return shared_memory[0];
}

__device__ __forceinline__ float block_reduce_max(
    float value,
    float* shared_memory
) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    // Stage 1: reduce independently inside all eight warps.
    value = warp_reduce_max(value);

    if (lane_id == 0) {
        shared_memory[warp_id] = value;
    }
    __syncthreads();

    // Stage 2: warp zero reduces the eight warp results. Negative infinity is
    // the identity for max, including when every real value is negative.
    if (warp_id == 0) {
        value =
            (lane_id < NUM_WARPS)
            ? shared_memory[lane_id]
            : -INFINITY;

        value = warp_reduce_max(value);

        if (lane_id == 0) {
            shared_memory[0] = value;
        }
    }
    __syncthreads();

    return shared_memory[0];
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}