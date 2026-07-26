#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "config.cuh"


extern "C" cudaError_t launch_decode_kernel(
    int token_id,
    const void* embed_weight,
    const void* lm_head_weight,
    const void* final_norm_weight,
    const void* layer_weights,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_residual,
    void* g_activations,
    void* g_attn_output,
    void* g_normalized,
    void* g_mlp_intermediate,
    void* g_logits,
    void* g_q,
    void* g_k,
    void* g_v,
    int position,
    int cache_len,
    int max_seq_len,
    cudaStream_t stream
);


torch::Tensor decode(
    int64_t token_id,
    torch::Tensor embed_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor final_norm_weight,
    std::vector<torch::Tensor> layer_weights,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t position,
    int64_t max_seq_len
) {
    auto float_options = embed_weight.options().dtype(torch::kFloat32);

    auto hidden_buffer = torch::empty({HIDDEN_SIZE}, embed_weight.options());
    auto g_residual = torch::empty({HIDDEN_SIZE}, float_options);
    auto g_activations = torch::empty({HIDDEN_SIZE}, float_options);
    auto g_attn_output = torch::empty({Q_SIZE}, float_options);
    auto g_normalized = torch::empty({HIDDEN_SIZE}, float_options);
    auto g_mlp_intermediate = torch::empty({INTERMEDIATE_SIZE}, float_options);
    auto g_logits = torch::empty({VOCAB_SIZE}, float_options);
    auto g_q = torch::empty({Q_SIZE}, float_options);
    auto g_k = torch::empty({KV_SIZE}, float_options);
    auto g_v = torch::empty({KV_SIZE}, float_options);

    std::vector<LayerWeights> host_layers(NUM_LAYERS);

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        int i = layer * 11;

        host_layers[layer] = {
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 1].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 2].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 3].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 4].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 5].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 6].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 7].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 8].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 9].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(layer_weights[i + 10].data_ptr())
        };
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    LayerWeights* device_layers = nullptr;
    C10_CUDA_CHECK(cudaMalloc(
        &device_layers,
        NUM_LAYERS * sizeof(LayerWeights)
    ));

    C10_CUDA_CHECK(cudaMemcpyAsync(
        device_layers,
        host_layers.data(),
        NUM_LAYERS * sizeof(LayerWeights),
        cudaMemcpyHostToDevice,
        stream
    ));

    int cache_len = static_cast<int>(position) + 1;

    C10_CUDA_CHECK(launch_decode_kernel(
        static_cast<int>(token_id),
        embed_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        final_norm_weight.data_ptr(),
        device_layers,
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden_buffer.data_ptr(),
        g_residual.data_ptr(),
        g_activations.data_ptr(),
        g_attn_output.data_ptr(),
        g_normalized.data_ptr(),
        g_mlp_intermediate.data_ptr(),
        g_logits.data_ptr(),
        g_q.data_ptr(),
        g_k.data_ptr(),
        g_v.data_ptr(),
        static_cast<int>(position),
        cache_len,
        static_cast<int>(max_seq_len),
        stream
    ));

    C10_CUDA_CHECK(cudaFree(device_layers));

    return g_logits;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("decode", &decode, "Qwen3 single-token decode");
}
