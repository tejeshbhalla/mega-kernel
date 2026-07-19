from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).parent

# PyTorch compiles megakernel.cu and the Python binding into one importable module.
kernel = load(
    name="megakernel_extension",
    sources=[
        str(ROOT / "megakernel.cu"),
        str(ROOT / "pytorch_binding.cu"),
    ],
    extra_include_paths=[str(ROOT)],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)


VOCAB_SIZE = 1000
HIDDEN_SIZE = 1024
TOKEN_ID = 42

embedding_weight = torch.randn(
    VOCAB_SIZE,
    HIDDEN_SIZE,
    device="cuda",
    dtype=torch.bfloat16,
)

# Our cooperative CUDA kernel.
kernel_output = kernel.embedding_lookup(embedding_weight, TOKEN_ID)

# The same operation using normal PyTorch indexing.
pytorch_output = embedding_weight[TOKEN_ID]

torch.cuda.synchronize()

print("kernel output: ", kernel_output[:10])
print("pytorch output:", pytorch_output[:10])
print("exact match:    ", torch.equal(kernel_output, pytorch_output))
