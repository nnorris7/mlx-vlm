#!/usr/bin/env python3
"""
A/B test TurboQuant norm correction.

Compares norm_correction=off vs norm_correction=on at several bit widths
using KL divergence vs unquantized baseline on WikiText-2.

Usage:
    .venv/bin/python scripts/turbo_norm_correction_ab.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --context-length 1024 --max-tokens 16384
"""

import argparse
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset

from mlx_vlm import load
from mlx_vlm.generate import maybe_quantize_kv_cache


def forward_logits(model, window_tokens, kv_bits, norm_correction):
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
            turbo_norm_correction=norm_correction,
        )

    out = model(input_ids, cache=cache)
    log_probs = nn.log_softmax(out.logits.astype(mx.float32), axis=-1)
    mx.eval(log_probs)
    del out, cache
    return log_probs[0]


def compare_variants(baseline_log_probs, variant_log_probs):
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
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--bit-configs",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 4.0],
        help="Bit widths to test",
    )
    args = parser.parse_args()

    print("=== TurboQuant Norm Correction A/B Test ===", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Bit configs: {args.bit_configs}", file=sys.stderr)

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

    # Accumulate: {(bits, nc): {kl, top1_agree, top10_agree}}
    results: dict = {}
    for bits in args.bit_configs:
        for nc in (False, True):
            results[(bits, nc)] = {"kl": 0.0, "top1_agree": 0.0, "top10_agree": 0.0}
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
        baseline_lp = forward_logits(model, window, None, False)
        print(f"  baseline: {time.time() - t0:.1f}s", file=sys.stderr)

        for bits in args.bit_configs:
            for nc in (False, True):
                t0 = time.time()
                var_lp = forward_logits(model, window, bits, nc)
                m = compare_variants(baseline_lp, var_lp)
                elapsed = time.time() - t0

                for k, v in m.items():
                    results[(bits, nc)][k] += v * args.context_length

                label = f"bits={bits} nc={int(nc)}"
                print(
                    f"  {label}: KL={m['kl']:.4f}  "
                    f"top1={m['top1_agree']:.4f}  "
                    f"top10={m['top10_agree']:.4f}  ({elapsed:.1f}s)",
                    file=sys.stderr,
                )

                del var_lp
                mx.clear_cache()

        n_positions_total += args.context_length
        del baseline_lp
        mx.clear_cache()

    print(f"\n=== Results ({n_positions_total} positions) ===")
    print(f"{'Config':<20} {'KL':>10} {'Top-1':>10} {'Top-10':>10}")
    print("-" * 55)
    for bits in args.bit_configs:
        for nc in (False, True):
            totals = results[(bits, nc)]
            kl = totals["kl"] / n_positions_total
            t1 = totals["top1_agree"] / n_positions_total
            t10 = totals["top10_agree"] / n_positions_total
            label = f"bits={bits} nc={'on' if nc else 'off'}"
            print(f"{label:<20} {kl:>10.4f} {t1:>10.4f} {t10:>10.4f}")

    print()
    print("Lower KL = closer to baseline")


if __name__ == "__main__":
    main()
