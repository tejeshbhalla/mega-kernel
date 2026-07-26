#include "config.cuh"
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;


__device__ void proj_lm_head(
    const __nv_bfloat16 * __restrict__ lm_head_weight, //shape is vocab_size,1024
    const float * __restrict__ g_normalized, //shape is 1024
    float * __restrict__ g_logits //shape is vocab_size

){

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int global_warp_id = blockIdx.x * NUM_WARPS + warp_id;

    const int total_warps  = gridDim.x * NUM_WARPS;

    for (int row_id=global_warp_id;row_id<VOCAB_SIZE;row_id+=total_warps){
        const __nv_bfloat16 * lm_head_row = lm_head_weight + static_cast<size_t>(row_id) * HIDDEN_SIZE; //right row 

        float partial_sum = 0.0f;

        for (int dimension=lane_id;dimension<HIDDEN_SIZE;dimension+=WARP_SIZE){

            partial_sum += __bfloat162float(lm_head_row[dimension]) * g_normalized[dimension];


        }

        const float full_sum = warp_reduce_sum(partial_sum);


        if (lane_id==0){

            g_logits[row_id] = full_sum;

        }



    }


}




__device__ void  down_proj_residual(
    const __nv_bfloat16 * __restrict__ down_proj_weight, //shape is 1024,3072
    const float * __restrict__ g_mlp_intermediate, //shape is 3072 
    float * __restrict__ g_residual //shape is 1024,

){

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int global_warp_id = blockIdx.x * NUM_WARPS + warp_id;

    const int total_warps  = gridDim.x * NUM_WARPS;

    for (int row_id=global_warp_id;row_id<HIDDEN_SIZE;row_id+=total_warps){

        const __nv_bfloat16 *down_row_data = down_proj_weight + static_cast<size_t>(row_id) * INTERMEDIATE_SIZE;

        float partial_sum = 0.0f;

        for (int dimension=lane_id;dimension<INTERMEDIATE_SIZE;dimension+=WARP_SIZE){

            partial_sum+= __bfloat162float(down_row_data[dimension]) * g_mlp_intermediate[dimension];

        }

        float full_row_sum = warp_reduce_sum(partial_sum);

        if (lane_id==0){

            g_residual[row_id] += full_row_sum;

        }


    }


}









__device__ void  fused_gate_up_silu(
    const __nv_bfloat16* __restrict__ gate_proj_weight, 
    const __nv_bfloat16* __restrict__ up_proj_weight,

    const float * __restrict__ g_normalized,
    float * __restrict__ g_mlp_intermediate //stores 3072 intermediate silu(gate) * up values 
){

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int global_warp_id = blockIdx.x * NUM_WARPS + warp_id;

    const int total_warps  = gridDim.x * NUM_WARPS;

    for (int row_id=global_warp_id;row_id<INTERMEDIATE_SIZE;row_id+=total_warps){

        const __nv_bfloat16 *gate_row = gate_proj_weight + (static_cast<size_t>(row_id) * HIDDEN_SIZE);
        const __nv_bfloat16 *up_row = up_proj_weight + (static_cast<size_t>(row_id) * HIDDEN_SIZE);

        float partial_gate_sum = 0.0f;
        float partial_up_sum = 0.0f;

        for (int dimension = lane_id;dimension<HIDDEN_SIZE;dimension+=WARP_SIZE){

            partial_gate_sum += __bfloat162float(gate_row[dimension]) * g_normalized[dimension];

            partial_up_sum += __bfloat162float(up_row[dimension]) * g_normalized[dimension];
            
        }

        const float full_row_gate_sum = warp_reduce_sum(partial_gate_sum);
        const float full_row_up_sum = warp_reduce_sum(partial_up_sum);



        if (lane_id==0){
            g_mlp_intermediate[row_id] = silu(full_row_gate_sum) * full_row_up_sum;

        }

    }

}





__device__ void post_attn_rms_norm(
    const float * __restrict__ g_residual,
    const __nv_bfloat16 * __restrict__ post_attn_norm_weight,
    float * __restrict__ g_normalized,
    float* smem

){

    if (blockIdx.x!=0){
        return ;
    } //rms norm on a flat 1024 vector doesnt need more blocks we will reduce accross one blocks 256 threads 

    const int thread_id = threadIdx.x;

    float local_sum_sq = 0.0f;

    for (int i=thread_id;i<HIDDEN_SIZE;i+=BLOCK_SIZE){
        local_sum_sq+= g_residual[i] * g_residual[i];
    }

    float total_sum = block_reduce_sum(
        local_sum_sq,
        smem
    );

    total_sum = total_sum / static_cast<float>(HIDDEN_SIZE);

    float rms_value  = rsqrtf(total_sum + RMS_NORM_EPS);

    for (int i=thread_id;i<HIDDEN_SIZE;i+=BLOCK_SIZE){
        g_normalized[i] = g_residual[i] * __bfloat162float(post_attn_norm_weight[i]) * rms_value;
    }


}








__device__ void o_proj_residual(
    const float * __restrict__ g_attn_output,
    const __nv_bfloat16 * __restrict__ o_proj_weight,
    float * __restrict__ g_residual
){

    //estimate warp id and also estimate laneid and global warp id 

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int global_warp_id = blockIdx.x * NUM_WARPS + warp_id;

    const int total_warps  = gridDim.x * NUM_WARPS; //each warp does multiple rows that is 1024/total_warps

    for (int row_id=global_warp_id; row_id<HIDDEN_SIZE; row_id+=total_warps){

        int offset_o_proj= row_id * Q_SIZE;

        float partial_sum = 0.0f;

        for (int d=lane_id; d<Q_SIZE; d+=WARP_SIZE){

            partial_sum+= __bfloat162float(o_proj_weight[offset_o_proj+d]) * g_attn_output[d];
        
        }

        float full_sum = warp_reduce_sum(partial_sum);

        if (lane_id==0){
            g_residual[row_id] += full_sum;
        }
    }


}




__device__ void attention_decode(
    const float * __restrict__ g_q,
    const int layer_id,
    const  __nv_bfloat16 * __restrict__ k_cache,
    const __nv_bfloat16 * __restrict__ v_cache,
    float * __restrict__ g_attn_output, // attn output stores 16 heads 128 vals ,
    int cache_len,
    int max_seq_len

){
    //constants for processing 

    constexpr int REPEAT_KV = NUM_Q_HEADS/NUM_KV_HEADS; 
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int global_warp_id = blockIdx.x * NUM_WARPS + warp_id; //tells us what exact head we are in for the q and kv globally 
    //each warp does one head we have roughlt 16 heads so almost 2 blocks each of 8 warps sort this out 

    if (global_warp_id >= NUM_Q_HEADS){
        return;
    }

    const int q_head = global_warp_id;
    const int kv_head = global_warp_id / REPEAT_KV; //each q_head uses two kv_heads 0-0,1-0,2/2-1 so 2 uses 1 like this 

    const int q_offset = q_head * HEAD_DIM;
    const int VALUES_PER_LANE = HEAD_DIM / WARP_SIZE; //128/32 =4 

    float output_accumalator[VALUES_PER_LANE]; //stores per thread local register

    for (int i=0;i<VALUES_PER_LANE;i++){
        output_accumalator[i] = 0.0f;
    }

    float RUNNING_MAX = -INFINITY;
    float RUNNING_SUM = 0.0f;
    const float SCALE = rsqrtf(static_cast<float>(HEAD_DIM));


    /*
    This loop for this query needs to go to every key in the head we decided per thread as usual since we have 32 here 
    cache is shaped 
    num_heads,max_seq_len,head_dim
    */

    for (int key_value_dim=0;key_value_dim<cache_len;key_value_dim+=1){

        const size_t cache_offset = static_cast<size_t>(layer_id) * NUM_KV_HEADS * max_seq_len * static_cast<size_t>(HEAD_DIM) + static_cast<size_t>(kv_head) * max_seq_len * static_cast<size_t>(HEAD_DIM) + key_value_dim * static_cast<size_t>(HEAD_DIM);

        //so this moves us to right key from key 0 in the designated head we are in so moves us to
        //right head and key 

        float partial_score = 0.0f; //this stores product of q.k 

        for (int i=0;i<VALUES_PER_LANE;i++){
            const int dimension = lane_id + i * WARP_SIZE;
            const float q_value = g_q[q_offset+dimension];
            const float k_value  = __bfloat162float(k_cache[cache_offset+dimension]);

            partial_score += q_value * k_value;
        }

        float attention_score = warp_reduce_sum(partial_score); //warp reduces accorss 32 threads 

        attention_score = __shfl_sync(
            0xffffffff,
            attention_score,
            0
        ); //__shfl_sync works as passing (hex of all threads,value to be broadcasted , and index )
        

        attention_score *= SCALE;

        float NEW_MAX = fmaxf(RUNNING_MAX,attention_score);
        
        float correction_factor = expf(RUNNING_MAX-NEW_MAX);

        float current_val = expf(attention_score-NEW_MAX);

        RUNNING_SUM = correction_factor * RUNNING_SUM + current_val;

        //now each lane has the q.k score we will do values 
        //each thread does 4 values scale by the wight of query and keys sim for that value token 

        for (int i=0;i<VALUES_PER_LANE;i++){

            const int dimension = lane_id + i * WARP_SIZE;
            
            const float v_value = __bfloat162float(v_cache[cache_offset+dimension]);

            output_accumalator[i] = output_accumalator[i] * correction_factor + v_value * current_val;

        }

        RUNNING_MAX = NEW_MAX;


    }


    const float INVERSE_SUM = 1/RUNNING_SUM;

    for (int i=0;i<VALUES_PER_LANE;i++){

            const int dimension = lane_id + i * WARP_SIZE;
            
            g_attn_output[q_offset+dimension] = output_accumalator[i] * INVERSE_SUM;

    }




}


__device__ void  qk_norm_rope_cache(
    float * __restrict__ g_q,
    float * __restrict__ g_k,
    float * __restrict__ g_v,

    int layer_id,

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

        const size_t cache_offset = static_cast<size_t>(layer_id)* NUM_KV_HEADS * max_seq_len * HEAD_DIM + static_cast<size_t>(kv_head) * max_seq_len * HEAD_DIM + static_cast<size_t>(position) * HEAD_DIM; //strides since kv cache is in pattern of kv_heads,max_seq_len,head_dim , you need to stride in memory kv_head*max_seq_len*head_dim in order to get to right positon for right kv head then for correct token its position + HEAD_DIM


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


__device__ void qkv_projection(
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    const float * g_activations,
    float * g_q,
    float * g_k,
    float * g_v

){

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

}


__device__ void input_rms_norm(
    const __nv_bfloat16 * norm_weight,
    const float * g_residual,
    float * smem,
    float * g_activations

){


    if (blockIdx.x==0){
                    //in this kernel we do rms norm and populate   g_activations 

        float local_sum_sq = 0.0f;

        for (int i=threadIdx.x;i<HIDDEN_SIZE;i+=BLOCK_SIZE){
                float value  = g_residual[i];
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

}




__global__ void decode_kernel(
    int token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const __nv_bfloat16* __restrict__ lm_head_weight,
    const __nv_bfloat16* __restrict__ final_norm_weight,

    const LayerWeights* __restrict__ layer_weights,

    
    const __nv_bfloat16* __restrict__ cos_table, 
    const __nv_bfloat16* __restrict__ sin_table, 

    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,

    __nv_bfloat16* __restrict__ hidden_buffer,
    float * __restrict__ g_residual,
    float * __restrict__ g_activations,
    float * __restrict__ g_attn_output,
    float * __restrict__ g_normalized,
    float * __restrict__ g_mlp_intermediate,
    float * __restrict__ g_logits,

    float * __restrict__ g_q,
    float * __restrict__ g_k,
    float * __restrict__ g_v,

    int position,
    int cache_len,
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
        g_residual[i] = __bfloat162float(row_ptr[i]);
    }

    grid.sync();

    //happens outside layer because we do embed_weight only once 

    for (int layer_idx=0; layer_idx<NUM_LAYERS;layer_idx+=1){

        const LayerWeights *lw = &layer_weights[layer_idx];

        input_rms_norm(lw->norm_weight,g_residual,smem,g_activations); 
     
        grid.sync();


        qkv_projection(lw->q_weight,lw->k_weight,lw->v_weight,g_activations,g_q,g_k,g_v);

        grid.sync();  //till here we have q,k,v vectors now we have to do q,k norm and rope 


        qk_norm_rope_cache(
                g_q,g_k,g_v,layer_idx,lw->q_norm_weight,lw->k_norm_weight,cos_table,sin_table,k_cache,v_cache,position,max_seq_len
        );

        //update cache len 
        
        grid.sync();

            

        attention_decode(g_q,layer_idx,k_cache,v_cache,g_attn_output,cache_len,max_seq_len);
        //kernel till here does attention decode and fills out g_attn_output of size n_q_heads*head_dim 

        grid.sync();

        //fused kernel for doing o_proj and doing skip connection/ residual add 


        o_proj_residual(
                g_attn_output,
                lw->o_proj_weight,
                g_residual
        );
            
        grid.sync();

        post_attn_rms_norm(
                g_residual,
                lw->post_attn_norm_weight,
                g_normalized,
                smem
        );

        grid.sync();


        fused_gate_up_silu(
                lw->gate_proj_weight,
                lw->up_proj_weight,
                g_normalized,
                g_mlp_intermediate
        ); //g_mlp_intermediate is up_projected to 3072 dims 

        grid.sync();

        down_proj_residual(
                lw->down_proj_weight,
                g_mlp_intermediate,
                g_residual
        );

        grid.sync();

}

    //all layers are done here now we just do post attention rms norm

    post_attn_rms_norm(
                g_residual,
                final_norm_weight,
                g_normalized,
                smem
        );

    grid.sync();
    //at this point g_normalized has the final output of the model of size 1024 

    //we need to proj it with lm_head_weight to get logits of size vocab_size in a intermediate buffer with size vocab_size 

    proj_lm_head(
        lm_head_weight,
        g_normalized,
        g_logits
    );

    grid.sync();




}




extern "C" cudaError_t launch_decode_kernel(
    int token_id,

    const void * embed_weight,
    const void * lm_head_weight,
    const void * final_norm_weight,

    const void * layer_weights,

    const void * cos_table,
    const void * sin_table,

    void * k_cache,
    void * v_cache,
    void * hidden_buffer,

    void *  g_residual,
    void *  g_activations,
    void *  g_attn_output,
    void *  g_normalized,
    void *  g_mlp_intermediate,
    void *  g_logits,
    void *  g_q,
    void *  g_k,
    void *  g_v,

    int position,
    int cache_len,
    int max_seq_len,
    cudaStream_t stream)
{
    const __nv_bfloat16 *typed_embed_weight =
        static_cast<const __nv_bfloat16*>(embed_weight);
    const __nv_bfloat16 *typed_lm_head_weight =
        static_cast<const __nv_bfloat16*>(lm_head_weight);
    const __nv_bfloat16 *typed_final_norm_weight =
        static_cast<const __nv_bfloat16*>(final_norm_weight);

    const LayerWeights * typed_layer_weights = static_cast<const LayerWeights *>(layer_weights);

    const __nv_bfloat16 *typed_cos_table =
        static_cast<const __nv_bfloat16*>(cos_table);
    const __nv_bfloat16 *typed_sin_table =
        static_cast<const __nv_bfloat16*>(sin_table);
    __nv_bfloat16 *typed_k_cache =
        static_cast<__nv_bfloat16*>(k_cache);
    __nv_bfloat16 *typed_v_cache =
        static_cast<__nv_bfloat16*>(v_cache);


    __nv_bfloat16 *typed_hidden_buffer =
        static_cast<__nv_bfloat16*>(hidden_buffer);
    float *typed_g_residual = static_cast<float*>(g_residual);
    float *typed_g_activations = static_cast<float*>(g_activations);
    float *typed_g_attn_output = static_cast<float*>(g_attn_output);
    float *typed_g_normalized = static_cast<float*>(g_normalized);
    float *typed_g_mlp_intermediate = static_cast<float*>(g_mlp_intermediate);
    float *typed_g_logits = static_cast<float*>(g_logits);
    float *typed_g_q = static_cast<float*>(g_q);
    float *typed_g_k = static_cast<float*>(g_k);
    float *typed_g_v = static_cast<float*>(g_v);


    void* kernel_args[] = {
        &token_id,
        &typed_embed_weight,
        &typed_lm_head_weight,
        &typed_final_norm_weight,
        &typed_layer_weights,
        &typed_cos_table,
        &typed_sin_table,
        &typed_k_cache,
        &typed_v_cache,
        &typed_hidden_buffer,
        &typed_g_residual,
        &typed_g_activations,
        &typed_g_attn_output,
        &typed_g_normalized,
        &typed_g_mlp_intermediate,
        &typed_g_logits,
        &typed_g_q,
        &typed_g_k,
        &typed_g_v,
        &position,
        &cache_len,
        &max_seq_len
    };

    return cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(decode_kernel),  //convert to  void * from void because this takes pointer ref to function
        dim3(DECODE_NUM_BLOCKS),
        dim3(BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );
}
