#include "config.cuh"
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

__global__ void embedding_lookup_kernel(
    int token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,


    __nv_bfloat16* __restrict__ hidden_buffer,
    float * __restrict__ g_residual,
    float * __restrict__ g_activations,

    float * __restrict__ g_q,
    float * __restrict__ g_k,
    float * __restrict__ g_v
)
{
    cg::grid_group grid = cg::this_grid();
    
    // shared warp memory 
    __shared__ float smem[NUM_WARPS]; //memory with 8 slots 


    const __nv_bfloat16* row_ptr =
        embed_weight + static_cast<size_t>(token_id) * HIDDEN_SIZE;

    const int global_thread_id =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int total_threads =
        gridDim.x * blockDim.x;

    for (int i = global_thread_id; i < HIDDEN_SIZE; i += total_threads) {
        hidden_buffer[i] = row_ptr[i];
    }

    grid.sync();


    if (blockIdx.x==0){
        //in this kernel we do rms norm and populate g_residual and g_activations 

        float local_sum_sq = 0.0f;

        for (int i=threadIdx.x;i<HIDDEN_SIZE;i+=BLOCK_SIZE){
            float value  = __bfloat162float(hidden_buffer[i]);

            g_residual[i] = value;

            local_sum_sq +=  value * value;
        }

        //here we have partial sums accorss the threads in this block we have 32 threads in 8 warps 
        
        float total_sum_sq = block_reduce_sum(local_sum_sq,smem);

        float mean_sum_sq = total_sum_sq / (HIDDEN_SIZE);

        float rms = rsqrtf(mean_sum_sq + RMS_NORM_EPS);

        for (int i=threadIdx.x;i<HIDDEN_SIZE;i+=BLOCK_SIZE){
            float value  = g_residual[i];
            float weight = __bfloat162float(norm_weight[i]);

            float rms_norm_output = value * weight * rms;

            g_activations[i] = rms_norm_output;
        }


    }

    grid.sync();


    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;

    int warp_id  = threadIdx.x / WARP_SIZE;
    int lane_id  = threadIdx.x % WARP_SIZE;

    int global_warp_id = blockIdx.x * NUM_WARPS + warp_id; //this tells us the row we wanna use 

    int total_warps = gridDim.x * NUM_WARPS;

    for (int row_id=global_warp_id;row_id<TOTAL_ROWS;row_id+=total_warps){

        const __nv_bfloat16 * weight_row;
        float * output_element;


        if (row_id<Q_SIZE){
            int q_row = row_id;

            weight_row = q_weight + q_row * HIDDEN_SIZE;
            output_element = g_q + q_row; //Question 
        } else if (row_id< Q_SIZE + KV_SIZE){

            int k_row = row_id - Q_SIZE;

            weight_row = k_weight + k_row * HIDDEN_SIZE;

            output_element = g_k + k_row;

 
        } else {

            int v_row = row_id - Q_SIZE - KV_SIZE;

            weight_row = v_weight + v_row * HIDDEN_SIZE;

            output_element = g_v + v_row;


        }

        float partial_sum = 0.0f;

        for (int k=lane_id;k<HIDDEN_SIZE;k+=WARP_SIZE){
            float weight = __bfloat162float(weight_row[k]);
            partial_sum += weight * g_activations[k];

        }

        partial_sum = warp_reduce_sum(partial_sum); //reduces sum for that row of q,k or v we got for 32 threads each lane doing skip of 32 elements doing 1024/32 elements each so partial sum only has sum for that thread we do warp reduce accorss 32 threads ot get full row output

        if (lane_id==0){
            *output_element = partial_sum;  //deref pointer and store value at correct position since gq is of shape 2048,1
        } 


    }


    grid.sync();


}

extern "C" cudaError_t launch_embedding_lookup(
    int token_id,
    const void* embed_weight,
    void* hidden_buffer,
    cudaStream_t stream)
{
    const __nv_bfloat16 *typed_embed_weight =
        static_cast<const __nv_bfloat16*>(embed_weight);

    __nv_bfloat16 *typed_hidden_buffer =
        static_cast<__nv_bfloat16*>(hidden_buffer);

    void* kernel_args[] = {
        &token_id,
        &typed_embed_weight,
        &typed_hidden_buffer
    };

    return cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(embedding_lookup_kernel),
        dim3(DECODE_NUM_BLOCKS),
        dim3(BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );
}