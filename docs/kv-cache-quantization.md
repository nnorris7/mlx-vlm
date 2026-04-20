# KV Cache Quantization

mlx-vlm supports two KV cache quantization backends: **uniform affine** (the
MLX default) and **TurboQuant** (rotation-based vector quantization with
optimal scalar codebooks). Both reduce memory usage during long-context
generation, trading a small quality loss for significantly smaller caches.

## Quick reference

| Flag | Default | Purpose |
|------|---------|---------|
| `--kv-bits N` | none | Total bit-width (triggers quantization) |
| `--kv-quant-scheme {uniform,turboquant}` | `turboquant` | Backend to use |
| `--kv-bits-k N` | none | Explicit key-cache bits (TurboQuant only) |
| `--kv-bits-v N` | none | Explicit value-cache bits (TurboQuant only) |
| `--turbo-boundary-layers N` | 0 | Leave first N + last N full-attn layers unquantized |
| `--turbo-norm-correction` | off | Store corrected norm for better quality (opt-in) |
| `--turbo-sparse-v` | off | Skip value codebook lookup when softmax weight < 1e-6 (long contexts) |
| `--quantized-kv-start N` | 5000 | Token count at which quantization activates |
| `--kv-group-size N` | 64 | Group size for uniform scheme (ignored by TurboQuant) |
| `--max-kv-size N` | none | Hard cap on KV cache size in tokens |

All flags work identically in `mlx_vlm generate` and `mlx_vlm.server` (the
server also accepts environment variables — see below).

---

## Quickest path

For most users running a VLM over long contexts, just pass `--kv-bits 4`:

```bash
mlx_vlm generate \
    --model mlx-community/gemma-4-26b-a4b-it-4bit \
    --image photo.jpg \
    --prompt "Describe this image" \
    --kv-bits 4 \
    --max-tokens 1024
```

This uses TurboQuant at 4 bits (keys and values both at 4 bits), which is
near-lossless for most tasks. KV cache memory drops by roughly 4×.

---

## Fractional bits (asymmetric K/V)

If you specify a fractional `--kv-bits` value (e.g. `3.5`), TurboQuant
automatically splits it across K and V: **keys get `floor`, values get `ceil`**.

```bash
--kv-bits 3.5  # → K=3, V=4
--kv-bits 2.5  # → K=2, V=3
```

The comment in `turboquant.py` says *"Values benefit more from extra bits"* —
this is the direction that wins in practice at moderate bit widths. A/B testing
on Gemma 4 (see vault: *mlx-vlm Research Synthesis 2026-04-11*) confirmed this
direction is fine at 3.5+ bits.

---

## Explicit independent K/V bits

If you want to pick K and V bit-widths independently (either direction), use
`--kv-bits-k` and `--kv-bits-v`. **Must be used together** and requires
`--kv-quant-scheme turboquant`.

```bash
# Turney's direction (keys get more bits — useful at aggressive compression)
--kv-bits-k 3 --kv-bits-v 2 --kv-quant-scheme turboquant

# Standard direction (explicit)
--kv-bits-k 3 --kv-bits-v 4 --kv-quant-scheme turboquant
```

**When to use which direction:**

- At 3+ bits per component, either direction produces nearly identical quality.
  Stick with Prince's default (V gets more bits via `--kv-bits 3.5` etc.).
- At ≤2.5 bits per component, keys become more sensitive. If you're pushing
  aggressive compression, consider `--kv-bits-k 3 --kv-bits-v 2` to avoid
  catastrophic quality loss.

These flags override `--kv-bits` entirely — if you pass all three, the
explicit K/V values win.

---

## Boundary layers (layer-aware precision)

`--turbo-boundary-layers N` keeps the first N and last N **full-attention**
layers unquantized (FP16), only applying TurboQuant to the middle layers.
Based on Turney's finding that boundary layers disproportionately contribute
to output quality.

```bash
# Protect the first 2 and last 2 full-attention layers
--kv-bits 2 --turbo-boundary-layers 2 --kv-quant-scheme turboquant
```

**A/B test results on Gemma 4 26B at 2-bit TurboQuant** (KL divergence vs
unquantized baseline, lower is better):

| Config | KL | Top-1 agree |
|--------|-----|-------------|
| `--turbo-boundary-layers 0` | 1.74 | 60.27% |
| `--turbo-boundary-layers 1` | 1.26 | 63.95% |
| `--turbo-boundary-layers 2` | 0.59 | 75.15% |

### Caveat for Gemma 4 specifically

Gemma 4 uses sliding-window attention on 5 of every 6 layers. Only ~17% of
layers (the full-attention ones) go through the TurboQuant codec. For
gemma-4-26b that's 5 full-attention layers total, and for gemma-4-31b it's 6.

This means:

- `--turbo-boundary-layers 2` on Gemma 4 leaves only 1 layer quantized (of 5).
  Quality is excellent but you get almost no memory savings.
- `--turbo-boundary-layers 1` is the sweet spot for Gemma 4 — protects the
  first and last full-attention layers, still compresses 3 of 5 (60%).

For non-SWA models (Llama, Qwen, Mistral) with 32+ layers, `--turbo-boundary-layers 2`
leaves 28+ of 32 layers quantized (87.5% compression retained), which is much
better balanced.

Sliding-window layers are always kept unquantized regardless of this flag
(they're not yet in the TurboQuant codec path).

---

## Norm correction

`--turbo-norm-correction` stores `original_norm / recon_norm` instead of the
raw norm, where `recon_norm` is the L2 norm of the quantized-and-dequantized
vector. This guarantees the dequantized vector has the same L2 norm as the
original — only the direction is approximated.

```bash
--kv-bits 2 --kv-quant-scheme turboquant --turbo-norm-correction
```

**A/B test on Gemma 4 26B** (KL vs unquantized baseline):

| Bits | KL off | KL on | KL Δ | Top-1 off | Top-1 on |
|------|--------|-------|------|-----------|----------|
| 2.0  | 1.7399 | 1.5737 | **-9.6%** | 60.27% | **61.92%** |
| 3.0  | 0.9579 | 0.9455 | -1.3% | 71.06% | 71.18% |
| 4.0  | 0.6437 | 0.6094 | **-5.3%** | 76.82% | **77.27%** |

Small but consistent improvement across all bit widths, strongest at
aggressive compression where quantization error is largest. Costs one extra
dispatch per quantize. Off by default — opt in when you want a bit more
quality at minor CPU cost.

Credited to `@spiritbuun` via Tom Turney's TurboQuant+ research.

---

## Sparse V dequant

`--turbo-sparse-v` skips the value codebook lookup in the fused decode kernels
when the softmax weight is below 1e-6. The check is uniform across the
simdgroup (all 32 lanes process the same token), so there's no SIMD divergence
cost — just a single comparison per token.

```bash
--kv-bits 4 --kv-quant-scheme turboquant --turbo-sparse-v
```

**When it helps:** Long contexts (32K+) where attention is sparse and most
tokens contribute negligibly to any given query. Turney reports +22.8% decode
speed at 32K on MoE models. At short contexts (under ~4K tokens), attention
is dense and the optimization has no effect — but also no cost, since the
threshold check is essentially free.

**Correctness verified:** A/B testing confirms bit-identical output between
`sparse_v=on` and `sparse_v=off` across 2/3/4 bits — the 1e-6 threshold only
skips contributions that wouldn't change the result anyway. Safe to enable
unconditionally; it's only off by default to keep changes opt-in.

Affects both single-pass (≤2048 tokens) and 2-pass (>2048 tokens) decode
kernels.

---

## Scheme selection

```bash
--kv-quant-scheme uniform     # MLX affine quantization (legacy, default group_size=64)
--kv-quant-scheme turboquant  # Rotation-based + MSE codebook (recommended for VLMs)
```

If you pass a fractional `--kv-bits` value, TurboQuant is required (uniform
doesn't support fractional bits). The asymmetric K/V and boundary-layer flags
also require TurboQuant.

---

## When does quantization actually activate?

`--quantized-kv-start N` controls the token offset at which quantization kicks
in. **Default is 5000**, meaning short prompts/generations don't get quantized
at all. This is sensible because the bandwidth savings only matter at long
contexts.

If you want to force quantization on from the start (useful for A/B testing
or very aggressive compression on short prompts), pass `--quantized-kv-start 0`:

```bash
mlx_vlm generate \
    --model ... --kv-bits 4 --quantized-kv-start 0 \
    --prompt "short prompt"
```

**Gotcha:** if you pass `--kv-bits 4` on a short prompt and don't see any
memory reduction, check `--quantized-kv-start`. You're probably running
unquantized.

---

## Running the server with quantization

All the same flags work on the server, and each is also backed by an
environment variable:

```bash
mlx_vlm.server \
    --model mlx-community/gemma-4-26b-a4b-it-4bit \
    --kv-bits 4 \
    --kv-quant-scheme turboquant \
    --turbo-boundary-layers 1 \
    --host 0.0.0.0 --port 8080
```

Environment variable equivalents:

| Flag | Env var |
|------|---------|
| `--kv-bits` | `KV_BITS` |
| `--kv-bits-k` | `KV_BITS_K` |
| `--kv-bits-v` | `KV_BITS_V` |
| `--turbo-boundary-layers` | `TURBO_BOUNDARY_LAYERS` |
| `--turbo-norm-correction` | `TURBO_NORM_CORRECTION` (`1`/`0`) |
| `--turbo-sparse-v` | `TURBO_SPARSE_V` (`1`/`0`) |
| `--kv-quant-scheme` | `KV_QUANT_SCHEME` |
| `--kv-group-size` | `KV_GROUP_SIZE` |
| `--max-kv-size` | `MAX_KV_SIZE` |
| `--quantized-kv-start` | `QUANTIZED_KV_START` |
| `--prefill-step-size` | `PREFILL_STEP_SIZE` |

**Note:** Quantization-aware (QAT) models skip KV cache quantization
automatically — if the model name contains `qat`, the flag is ignored and
you'll see a warning.

---

## Common combinations

**Maximum quality, meaningful memory savings** (default-ish):
```bash
--kv-bits 4 --kv-quant-scheme turboquant
```

**Aggressive compression for very long contexts on Gemma 4:**
```bash
--kv-bits 3 --kv-quant-scheme turboquant --turbo-boundary-layers 1
```

**Extreme compression for non-SWA models (Llama/Qwen/Mistral):**
```bash
--kv-bits 2 --kv-quant-scheme turboquant --turbo-boundary-layers 2
```

**Asymmetric for pushing quality at low bit-widths:**
```bash
--kv-bits-k 3 --kv-bits-v 2 --kv-quant-scheme turboquant --turbo-boundary-layers 2
```

**Force quantization on short prompts (testing):**
```bash
--kv-bits 4 --kv-quant-scheme turboquant --quantized-kv-start 0
```

**Long-context use (32K+) with maximum perf:**
```bash
--kv-bits 4 --kv-quant-scheme turboquant --turbo-sparse-v --turbo-norm-correction
```
