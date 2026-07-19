#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "config.cuh"


// This function is implemented in megakernel.cu.
extern "C" cudaError_t launch_embedding_lookup(
    int token_id,
    const void* embed_weight,
    void* hidden_buffer,
    cudaStream_t stream
);


torch::Tensor embedding_lookup(
    torch::Tensor embed_weight,
    int64_t token_id
) {
    torch::Tensor hidden_buffer = torch::empty(
        {HIDDEN_SIZE},
        embed_weight.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    C10_CUDA_CHECK(launch_embedding_lookup(
        static_cast<int>(token_id),
        embed_weight.data_ptr(),
        hidden_buffer.data_ptr(),
        stream
    ));

    return hidden_buffer;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "embedding_lookup",
        &embedding_lookup,
        "Cooperative embedding lookup"
    );
}
