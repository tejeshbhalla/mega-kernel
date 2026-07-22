#include "config.cuh"
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;


__device__ void  qk_norm_rope_cache(
    float * __restrict__ g_q,
    float * __restrict__ g_k,
    float * __restrict__ g_v,

    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table, 
    const __nv_bfloat16* __restrict__ sin_table, 
    __nv_bfloat16 * k_cache,
    __nv_bfloat16 * v_cache,
    int position,
    int max_seq_len
    
){
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;

    const int global_warp_id = blockIdx.x * NUM_WARPS + warp_id; //this tells us that globally what warp we are in 108*8

    const __nv_bfloat16  *cos_row = cos_table + static_cast<size_t>(position) * HEAD_DIM; 
    const __nv_bfloat16 *sin_row = sin_table + static_cast<size_t>(position) * HEAD_DIM;


    if (global_warp_id<NUM_Q_HEADS){

        const int q_head = global_warp_id;
        const int head_offset  = q_head * HEAD_DIM;

        float local_sum_sq = 0.0f;

        for (int d = lane_id;d<HEAD_DIM;d+=WARP_SIZE){
            const float value = g_q[head_offset+d];
            
            local_sum_sq += value * value;
        }

        float sum_sq  = warp_reduce_sum(local_sum_sq); //lane 0 has the final answer 

        sum_sq = __shfl_sync(0xffffffff,sum_sq,0); //broadcast lane id's 0 sum to all other lanes (the 0 in the end is what to broadcast from )

        const float rstd = rsqrtf(
            sum_sq / HEAD_DIM + RMS_NORM_EPS
        );

        for (int d = lane_id; d<HEAD_DIM/2;d+=WARP_SIZE){

            const int first_index  = head_offset +d;
            const int second_index = head_offset + d + HEAD_DIM /2;

            const float first_weight = __bfloat162float(q_norm_weight[d]); //weights is of shape HEAD_DIM , same accorss heads 
            const float second_weight  = __bfloat162float(q_norm_weight[d+HEAD_DIM/2]);

            const float first_value  = g_q[first_index] * rstd * first_weight;
            const float second_value  = g_q[second_index] * rstd * second_weight;


            //roatate each value now 
            const float cos_value  = __bfloat162float(cos_row[d]);
            const float sin_value = __bfloat162float(sin_row[d]);

            const float rotated_first  = first_value * cos_value - second_value * sin_value;

            const float rotated_second = second_value * cos_value + first_value * sin_value;

            g_q[first_index] = rotated_first;
            g_q[second_index] = rotated_second;

        } 




    } else if (global_warp_id < NUM_Q_HEADS + NUM_KV_HEADS)
    {

        const int kv_head = global_warp_id - NUM_Q_HEADS;
        const int head_offset  = kv_head * HEAD_DIM;

        float local_sum_sq = 0.0f;

        for (int d = lane_id;d<HEAD_DIM;d+=WARP_SIZE){
            const float value = g_k[head_offset+d];
            
            local_sum_sq += value * value;
        }

        float sum_sq  = warp_reduce_sum(local_sum_sq); //lane 0 has the final answer 

        sum_sq = __shfl_sync(0xffffffff,sum_sq,0); //broadcast lane id's 0 sum to all other lanes (the 0 in the end is what to broadcast from )

        const float rstd = rsqrtf(
            sum_sq / HEAD_DIM + RMS_NORM_EPS
        );

        const size_t cache_offset = static_cast<size_t>(kv_head) * max_seq_len * HEAD_DIM + static_cast<size_t>(position) * HEAD_DIM; //strides since kv cache is in pattern of kv_heads,max_seq_len,head_dim , you need to stride in memory kv_head*max_seq_len*head_dim in order to get to right positon for right kv head then for correct token its position + HEAD_DIM




        for (int d = lane_id; d<HEAD_DIM/2;d+=WARP_SIZE){

            const int first_index  = head_offset +d;
            const int second_index = head_offset + d + HEAD_DIM /2;

            const float first_weight = __bfloat162float(k_norm_weight[d]); //weights is of shape HEAD_DIM , same accorss heads 
            const float second_weight  = __bfloat162float(k_norm_weight[d+HEAD_DIM/2]);

            const float first_value  = g_k[first_index] * rstd * first_weight;
            const float second_value  = g_k[second_index] * rstd * second_weight;


            //roatate each value now 
            const float cos_value  = __bfloat162float(cos_row[d]);
            const float sin_value = __bfloat162float(sin_row[d]);

            const float rotated_first  = first_value * cos_value - second_value * sin_value;

            const float rotated_second = second_value * cos_value + first_value * sin_value;

            g_k[first_index] = rotated_first;
            g_k[second_index] = rotated_second;

            k_cache[cache_offset+d] = __float2bfloat16(rotated_first);
            k_cache[cache_offset+d+HEAD_DIM/2] = __float2bfloat16(rotated_second);

        } 

        for (int d = lane_id;d<HEAD_DIM;d+=WARP_SIZE){
            v_cache[cache_offset+d] = __float2bfloat16(g_v[head_offset+d]);
        }


    }
    
    




}






__global__ void embedding_lookup_kernel(
    int token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight, 

    const __nv_bfloat16* __restrict__ cos_table, 
    const __nv_bfloat16* __restrict__ sin_table, 

    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,

    __nv_bfloat16* __restrict__ hidden_buffer,
    float * __restrict__ g_residual,
    float * __restrict__ g_activations,

    float * __restrict__ g_q,
    float * __restrict__ g_k,
    float * __restrict__ g_v,

    int position,
    int max_seq_len
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
            partial_sum += weight * g_activations[k]; //every thread is doing indivdual partial sum with an offset of 32 

        }

        partial_sum = warp_reduce_sum(partial_sum); //reduces sum for that row of q,k or v we got for 32 threads each lane doing skip of 32 elements doing 1024/32 elements each so partial sum only has sum for that thread we do warp reduce accorss 32 threads ot get full row output

        if (lane_id==0){
            *output_element = partial_sum;  //deref pointer and store value at correct position since gq is of shape 2048,1
        } 


    }


    grid.sync();  //till here we have q,k,v vectors now we have to do q,k norm and rope 


    qk_norm_rope_cache(
        g_q,g_k,g_v,q_norm_weight,k_norm_weight,cos_table,sin_table,k_cache,v_cache,position,max_seq_len
    );

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


