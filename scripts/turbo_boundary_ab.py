#!/usr/bin/env python3
"""
A/B test TurboQuant boundary-layers at aggressive compression.

Compares three configurations against the unquantized baseline:
  1. Default (boundary=0): skip only the last layer (legacy)
  2. Boundary=2: skip first 2 + last 2 layers (Turney's finding)
  3. Boundary=4: skip first 4 + last 4 layers

Uses KL divergence, top-1 agreement, and top-10 agreement as metrics.

Usage:
    .venv/bin/python scripts/turbo_boundary_ab.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --kv-bits 2 --context-length 1024 --max-tokens 16384
"""

import argparse
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset

from mlx_vlm import load
from mlx_vlm.generate import maybe_quantize_kv_cache


def forward_logits(model, window_tokens, kv_bits, boundary_layers):
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
            kv_quant_scheme="turboquant",
            turbo_boundary_layers=boundary_layers,
        )

    out = model(input_ids, cache=cache)
    log_probs = nn.log_softmax(out.logits.astype(mx.float32), axis=-1)
    mx.eval(log_probs)
    del out, cache
    return log_probs[0]


def compare_variants(baseline_log_probs, variant_log_probs):
    """Compare a variant to baseline. Returns dict of metrics."""
    base = baseline_log_probs
    var = variant_log_probs

    base_probs = mx.exp(base)
    kl_per_position = (base_probs * (base - var)).sum(axis=-1)
    kl_mean = float(kl_per_position.mean())

    base_top1 = mx.argmax(base, axis=-1)
    var_top1 = mx.argmax(var, axis=-1)
    top1_agree = float((base_top1 == var_top1).astype(mx.float32).mean())

    var_top10 = mx.argsort(-var, axis=-1)[:, :10]
    base_top1_in_var_top10 = (var_top10 == base_top1[:, None]).any(axis=-1)
    top10_agree = float(base_top1_in_var_top10.astype(mx.float32).mean())

    return {"kl": kl_mean, "top1_agree": top1_agree, "top10_agree": top10_agree}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--kv-bits", type=float, default=2.0)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--boundary-configs",
        type=int,
        nargs="+",
        default=[0, 2, 4],
        help="Boundary-layer values to test",
    )
    args = parser.parse_args()

    print(f"=== TurboQuant Boundary Layers A/B Test ===", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"KV bits: {args.kv_bits}", file=sys.stderr)
    print(f"Boundary configs: {args.boundary_configs}", file=sys.stderr)

    print("\nLoading model...", file=sys.stderr)
    t0 = time.time()
    model, processor = load(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    tokenizer = processor.tokenizer

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    all_tokens = tokenizer.encode(text, add_special_tokens=False)[: args.max_tokens]
    bos_id = tokenizer.bos_token_id
    print(f"  {len(all_tokens)} tokens", file=sys.stderr)

    n_windows = len(all_tokens) // args.context_length
    if n_windows == 0:
        print("Not enough tokens for one window", file=sys.stderr)
        return

    variants = {
        f"boundary={b}": {"b": b, "totals": {"kl": 0.0, "top1_agree": 0.0, "top10_agree": 0.0}}
        for b in args.boundary_configs
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

        t0 = time.time()
        baseline_lp = forward_logits(model, window, None, 0)
        print(f"  baseline: {time.time() - t0:.1f}s", file=sys.stderr)

        for label, info in variants.items():
            t0 = time.time()
            var_lp = forward_logits(model, window, args.kv_bits, info["b"])
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

    print(f"\n=== Results ({n_positions_total} positions, kv_bits={args.kv_bits}) ===")
    print(f"{'Config':<15} {'KL':>10} {'Top-1':>10} {'Top-10':>10}")
    print("-" * 50)
    for label, info in variants.items():
        kl = info["totals"]["kl"] / n_positions_total
        t1 = info["totals"]["top1_agree"] / n_positions_total
        t10 = info["totals"]["top10_agree"] / n_positions_total
        print(f"{label:<15} {kl:>10.4f} {t1:>10.4f} {t10:>10.4f}")

    print()
    print("Lower KL = closer to baseline")
    print("Higher Top-1/Top-10 agreement = closer to baseline")


if __name__ == "__main__":
    main()
