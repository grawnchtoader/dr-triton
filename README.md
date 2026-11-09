# `dr-triton`: Fused Triton Dequantization & 3-Tier MoE Loader (`Strata-Coder`)

Official Triton & PyTorch runtime dequantization kernels and 3-tier MoE loader for:
- **[`Grawnch/Kimi-K3.1-Pro-Coder`](https://huggingface.co/Grawnch/Kimi-K3.1-Pro-Coder)** (`2.8T` MoE, 93 layers, **896 routed experts/layer** at **`2.81 / 2.56 / 1.44 bpw`**, `774.97 GB` total across 96 `.safetensors` shards)

Works alongside **[`grawnchtoader/strata`](https://github.com/grawnchtoader/strata)** for 3-tier residency management (**Tier 1 Hot GPU VRAM**, **Tier 2 Warm System RAM**, **Tier 3 Cold NVMe `mmap`** with `RouterLookahead` prefetching).

---

## Quickstart (`Kimi-K3.1-Pro-Coder`)

### 1. Install `dr-triton`
```bash
git clone https://github.com/grawnchtoader/dr-triton.git
cd dr-triton
pip install -e .
```

### 2. Start the Built-In Web GUI, Monitor Dashboard, or Interactive CLI
```bash
# Start built-in Strata Web GUI & Monitor Dashboard + OpenAI API server at http://127.0.0.1:8080:
python -m dr_triton.serve --model /path/to/Kimi-K3.1-Pro-Coder --port 8080

# Or run interactive terminal chat directly:
python -m dr_triton.serve --model /path/to/Kimi-K3.1-Pro-Coder --cli
```

### 3. Python API: Load a 3-Tier `Strata-Coder` MoE Layer
```python
from dr_triton import load_strata_coder_layer

# Automatically pins Tier 1 (top 15% / 134 coding/math experts) in VRAM,
# Tier 2 (next 30% / 269 warm experts) in System RAM, and streams Tier 3 (55% / 493 cold tail)
# from NVMe mmap with async RouterLookahead prefetching:
layer_cache = load_strata_coder_layer(
    shard_path="model-00002-of-000096.safetensors",
    layer_idx=1,
    device="cuda",
    vram_hot_budget_gb=0.8,
    ram_warm_budget_gb=2.5,
)

# Optional: asynchronously prefetch predicted Cold NVMe experts via RouterLookahead
layer_cache.prefetch_cold_experts_async([14, 102, 511])

# Decode expert (w1, w3, w2) on GPU
w1, w3, w2 = layer_cache.get_decoded_expert(14)
```

### 4. Multi-GPU + Dynamic RAM / V100 (`sm_70`) Auto-Support
`dr-triton` and `strata_coder` automatically detect your GPUs' compute capability (falling back to native `FP16` with `FP32` accumulation on `< sm_80` GPUs like Tesla V100 SXM2) and dynamically query your total GPU VRAM and Host System RAM, filling **Tier 1 → Tier 2 → Tier 3** in strict priority order:
```python
from dr_triton import load_strata_coder_multi_gpu

shard_paths = {layer_idx: f"model-{layer_idx+1:05d}-of-000096.safetensors" for layer_idx in range(1, 93)}
# Auto-detects GPU count, total VRAM, and total System RAM (or pass overrides if desired):
layer_managers = load_strata_coder_multi_gpu(shard_paths=shard_paths)
```
