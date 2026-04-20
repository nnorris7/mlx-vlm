#!/usr/bin/env python3
"""
A/B test TurboQuant K/V direction by measuring divergence from baseline.

Instead of absolute perplexity (unreliable on instruction-tuned models), this
computes how much each quantized variant's output distribution differs from
the unquantized baseline. Three metrics:

1. KL divergence (baseline || variant) — mean over all positions
2. Top-1 agreement — fraction of positions where argmax is unchanged
3. Top-10 agreement — fraction of positions where baseline's top-1 is in
   variant's top-10

Runs all three variants (baseline, V-more, K-more) in a single invocation
to ensure identical inputs.

Usage:
    .venv/bin/python scripts/turbo_ab_divergence.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --kv-bits 2.5 --context-length 1024 --max-tokens 8192
"""

import argparse
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset

from mlx_vlm import load
from mlx_vlm.generate import maybe_quantize_kv_cache


def forward_logits(model, window_tokens, kv_bits, kv_quant_scheme, boundary_layers=0):
    """Run a forward pass and return log-softmax logits [L, V]."""
    input_ids = mx.array(window_tokens)[None, :]
    lm = model.language_model if hasattr(model, "language_model") else model
    cache = lm.make_cache()

    if kv_bits is not None:
        maybe_quantize_kv_cache(
            cache,
            quantized_kv_start=0,
            kv_group_size=64,
            kv_bits=kv_bits,
            kv_quant_scheme=kv_quant_scheme,
            turbo_boundary_layers=boundary_layers,
        )

    out = model(input_ids, cache=cache)
    # Convert to log probs in float32 for stable math
    log_probs = nn.log_softmax(out.logits.astype(mx.float32), axis=-1)
    mx.eval(log_probs)
    del out, cache
    return log_probs[0]  # [L, V]


def compare_variants(baseline_log_probs, variant_log_probs):
    """Compare a variant to baseline. Returns dict of metrics."""
    # Skip last position (no target) — we compare position 0..L-2
    # (Though for divergence we're comparing full distributions, not targets)
    base = baseline_log_probs  # [L, V]
    var = variant_log_probs

    # KL(base || var) = sum_x base(x) * (log base(x) - log var(x))
    base_probs = mx.exp(base)
    kl_per_position = (base_probs * (base - var)).sum(axis=-1)  # [L]
    kl_mean = float(kl_per_position.mean())

    # Top-1 argmax agreement
    base_top1 = mx.argmax(base, axis=-1)  # [L]
    var_top1 = mx.argmax(var, axis=-1)
    top1_agree = float((base_top1 == var_top1).astype(mx.float32).mean())

    # Top-10 agreement: is baseline's top-1 in variant's top-10?
    var_top10 = mx.argsort(-var, axis=-1)[:, :10]  # [L, 10]
    base_top1_in_var_top10 = (var_top10 == base_top1[:, None]).any(axis=-1)
    top10_agree = float(base_top1_in_var_top10.astype(mx.float32).mean())

    return {
        "kl": kl_mean,
        "top1_agree": top1_agree,
        "top10_agree": top10_agree,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--kv-bits", type=float, default=2.5)
    parser.add_argument("--kv-quant-scheme", default="turboquant")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print(f"=== TurboQuant K/V Direction A/B Test ===", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"KV bits: {args.kv_bits}", file=sys.stderr)
    print(f"Context length: {args.context_length}", file=sys.stderr)

    # Load model once
    print("\nLoading model...", file=sys.stderr)
    t0 = time.time()
    model, processor = load(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    tokenizer = processor.tokenizer

    # Load dataset
    print(f"Loading {args.dataset}/{args.dataset_config} ({args.split})...", file=sys.stderr)
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    all_tokens = tokenizer.encode(text, add_special_tokens=False)[: args.max_tokens]
    bos_id = tokenizer.bos_token_id
    print(f"  {len(all_tokens)} tokens  bos_id={bos_id}", file=sys.stderr)

    n_windows = len(all_tokens) // args.context_length
    if n_windows == 0:
        print("Not enough tokens for even one window", file=sys.stderr)
        return

    # Accumulate metrics per variant
    variants = {
        "V-more (default)": {"k_more": False, "totals": {"kl": 0.0, "top1_agree": 0.0, "top10_agree": 0.0}},
        "K-more (Turney)": {"k_more": True, "totals": {"kl": 0.0, "top1_agree": 0.0, "top10_agree": 0.0}},
    }
    n_positions_total = 0

    print(f"\nRunning {n_windows} windows of {args.context_length} tokens...", file=sys.stderr)

    for win_idx in range(n_windows):
        start = win_idx * args.context_length
        end = start + args.context_length
        window = all_tokens[start:end]
        if bos_id is not None:
            window = [bos_id] + list(window[:-1])

        print(f"\n--- window {win_idx + 1}/{n_windows} ---", file=sys.stderr)

        # Baseline
        t0 = time.time()
        baseline_lp = forward_logits(model, window, None, args.kv_quant_scheme)
        print(f"  baseline: {time.time() - t0:.1f}s", file=sys.stderr)

        # Each variant
        for label, info in variants.items():
            if info["k_more"]:
                os.environ["TURBO_K_GETS_MORE_BITS"] = "1"
            else:
                os.environ["TURBO_K_GETS_MORE_BITS"] = "0"

            t0 = time.time()
            var_lp = forward_logits(model, window, args.kv_bits, args.kv_quant_scheme)
            metrics = compare_variants(baseline_lp, var_lp)
            elapsed = time.time() - t0

            for k, v in metrics.items():
                info["totals"][k] += v * args.context_length

            print(
                f"  {label}: KL={metrics['kl']:.4f}  "
                f"top1={metrics['top1_agree']:.4f}  "
                f"top10={metrics['top10_agree']:.4f}  "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
            )

            del var_lp
            mx.clear_cache()

        n_positions_total += args.context_length

        del baseline_lp
        mx.clear_cache()

    # Final averages
    print(f"\n=== Results ({n_positions_total} positions) ===")
    print(f"{'Variant':<25} {'KL':>10} {'Top-1':>10} {'Top-10':>10}")
    print("-" * 60)
    for label, info in variants.items():
        kl = info["totals"]["kl"] / n_positions_total
        t1 = info["totals"]["top1_agree"] / n_positions_total
        t10 = info["totals"]["top10_agree"] / n_positions_total
        print(f"{label:<25} {kl:>10.4f} {t1:>10.4f} {t10:>10.4f}")

    print()
    print("Lower KL = closer to baseline")
    print("Higher Top-1/Top-10 agreement = closer to baseline")
    print("Winner: whichever is better on more metrics")


if __name__ == "__main__":
    main()
