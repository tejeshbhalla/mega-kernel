from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 512


kernel = load(
    name="megakernel_extension",
    sources=[str(ROOT / "megakernel.cu"), str(ROOT / "pytorch_binding.cu")],
    extra_include_paths=[str(ROOT)],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
).cuda().eval()


layer_weights = []
for layer in model.model.layers:
    layer_weights.extend([
        layer.input_layernorm.weight,
        layer.self_attn.q_proj.weight,
        layer.self_attn.k_proj.weight,
        layer.self_attn.v_proj.weight,
        layer.self_attn.q_norm.weight,
        layer.self_attn.k_norm.weight,
        layer.self_attn.o_proj.weight,
        layer.post_attention_layernorm.weight,
        layer.mlp.gate_proj.weight,
        layer.mlp.up_proj.weight,
        layer.mlp.down_proj.weight,
    ])


dimensions = torch.arange(0, 128, 2, device="cuda", dtype=torch.float32)
inverse_frequency = 1.0 / (1_000_000.0 ** (dimensions / 128))
positions = torch.arange(MAX_SEQ_LEN, device="cuda", dtype=torch.float32)
frequencies = torch.outer(positions, inverse_frequency)
rope = torch.cat((frequencies, frequencies), dim=-1)
cos_table = rope.cos().to(torch.bfloat16)
sin_table = rope.sin().to(torch.bfloat16)


messages = [
    {"role": "user", "content": "Write para on ms dhoni?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
)
input_ids = inputs["input_ids"].cuda()
prompt_length = input_ids.shape[1]


# Hugging Face performs prompt prefill and creates its K/V cache.
with torch.inference_mode():
    output = model(input_ids, use_cache=True)

logits = output.logits[0, -1].float()
hf_cache = output.past_key_values


# Copy the Hugging Face cache into the layout expected by the CUDA kernel:
# [layer, kv_head, sequence, head_dimension].
cache_shape = (28, 8, MAX_SEQ_LEN, 128)
k_cache = torch.empty(cache_shape, device="cuda", dtype=torch.bfloat16)
v_cache = torch.empty_like(k_cache)

for layer in range(28):
    k_cache[layer, :, :prompt_length] = hf_cache.layers[layer].keys[0]
    v_cache[layer, :, :prompt_length] = hf_cache.layers[layer].values[0]


generated_tokens = []
decode_times = []

print("Assistant: ", end="", flush=True)

with torch.inference_mode():
    for position in range(prompt_length, MAX_SEQ_LEN):
        next_token = int(torch.argmax(logits).item())
        generated_tokens.append(next_token)

        print(tokenizer.decode([next_token]), end="", flush=True)

        if next_token == tokenizer.eos_token_id:
            break
        if len(generated_tokens) == MAX_NEW_TOKENS:
            break

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        logits = kernel.decode(
            next_token,
            model.model.embed_tokens.weight,
            model.lm_head.weight,
            model.model.norm.weight,
            layer_weights,
            cos_table,
            sin_table,
            k_cache,
            v_cache,
            position,
            MAX_SEQ_LEN,
        )
        end.record()

        end.synchronize()
        decode_times.append(start.elapsed_time(end))


kernel_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

if decode_times:
    average_ms = sum(decode_times) / len(decode_times)
    print(f"\nKernel measured tokens: {len(decode_times)}")
    print(f"Kernel decode time:     {average_ms:.3f} ms/token")
    print(f"Kernel throughput:      {1000.0 / average_ms:.2f} tokens/second")


# Run the same greedy decode with Hugging Face from a fresh prefill cache.
with torch.inference_mode():
    hf_output = model(input_ids, use_cache=True)

hf_logits = hf_output.logits[0, -1].float()
hf_cache = hf_output.past_key_values
hf_generated_tokens = []
hf_decode_times = []

with torch.inference_mode():
    for _ in range(MAX_NEW_TOKENS):
        next_token = int(torch.argmax(hf_logits).item())
        hf_generated_tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break
        if len(hf_generated_tokens) == MAX_NEW_TOKENS:
            break

        next_input = torch.tensor([[next_token]], device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        hf_output = model(
            next_input,
            past_key_values=hf_cache,
            use_cache=True,
        )
        end.record()

        end.synchronize()
        hf_decode_times.append(start.elapsed_time(end))
        hf_logits = hf_output.logits[0, -1].float()
        hf_cache = hf_output.past_key_values


hf_average_ms = sum(hf_decode_times) / len(hf_decode_times)
print(f"\nHF measured tokens:     {len(hf_decode_times)}")
print(f"HF decode time:         {hf_average_ms:.3f} ms/token")
print(f"HF throughput:          {1000.0 / hf_average_ms:.2f} tokens/second")
print(f"Kernel speedup:         {hf_average_ms / average_ms:.2f}x")

hf_text = tokenizer.decode(hf_generated_tokens, skip_special_tokens=True)

print("\n" + "=" * 70)
print("CUDA KERNEL GENERATED TEXT")
print("=" * 70)
print(kernel_text)

print("\n" + "=" * 70)
print("HUGGING FACE GENERATED TEXT")
print("=" * 70)
print(hf_text)
