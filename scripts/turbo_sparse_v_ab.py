#!/usr/bin/env python3
"""
A/B test TurboQuant sparse V dequant.

Compares sparse_v=off vs sparse_v=on at several bit widths. This is primarily
a CORRECTNESS check — sparse_v should produce essentially identical output
because we're only skipping contributions below 1e-6 (negligible).

The performance benefit is realized at long contexts (>32K tokens) where
attention is sparse. This script doesn't measure that — it just confirms
that sparse_v is mathematically equivalent at the bit widths and contexts
we can easily test.

Usage:
    .venv/bin/python scripts/turbo_sparse_v_ab.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --context-length 1024 --max-tokens 8192
"""

import argparse
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset

from mlx_vlm import load
from mlx_vlm.generate import maybe_quantize_kv_cache


def forward_logits(model, window_tokens, kv_bits, sparse_v):
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
            turbo_sparse_v=sparse_v,
        )
    out = model(input_ids, cache=cache)
    log_probs = nn.log_softmax(out.logits.astype(mx.float32), axis=-1)
    mx.eval(log_probs)
    del out, cache
    return log_probs[0]


def compare(base, var):
    base_probs = mx.exp(base)
    kl = float((base_probs * (base - var)).sum(axis=-1).mean())
    base_top1 = mx.argmax(base, axis=-1)
    var_top1 = mx.argmax(var, axis=-1)
    top1 = float((base_top1 == var_top1).astype(mx.float32).mean())
    var_top10 = mx.argsort(-var, axis=-1)[:, :10]
    top10 = float((var_top10 == base_top1[:, None]).any(axis=-1).astype(mx.float32).mean())
    return {"kl": kl, "top1": top1, "top10": top10}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--bit-configs", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    args = parser.parse_args()

    print("=== TurboQuant Sparse V A/B Test (Correctness) ===", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Bits: {args.bit_configs}", file=sys.stderr)

    print("\nLoading model...", file=sys.stderr)
    t0 = time.time()
    model, processor = load(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    tok = processor.tokenizer
    bos_id = tok.bos_token_id

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    all_tokens = tok.encode(text, add_special_tokens=False)[: args.max_tokens]
    print(f"  {len(all_tokens)} tokens", file=sys.stderr)

    n_windows = len(all_tokens) // args.context_length
    if n_windows == 0:
        return

    # Track per-config totals
    results = {}
    for bits in args.bit_configs:
        for sv in (False, True):
            results[(bits, sv)] = {"kl": 0.0, "top1": 0.0, "top10": 0.0}
    n_pos = 0

    print(f"\nRunning {n_windows} windows...", file=sys.stderr)
    for win_idx in range(n_windows):
        start = win_idx * args.context_length
        end = start + args.context_length
        window = all_tokens[start:end]
        if bos_id is not None:
            window = [bos_id] + list(window[:-1])

        print(f"\n--- window {win_idx + 1}/{n_windows} ---", file=sys.stderr)

        baseline = forward_logits(model, window, None, False)
        for bits in args.bit_configs:
            for sv in (False, True):
                t0 = time.time()
                v = forward_logits(model, window, bits, sv)
                m = compare(baseline, v)
                for k, val in m.items():
                    results[(bits, sv)][k] += val * args.context_length
                print(
                    f"  bits={bits} sparse_v={int(sv)}: "
                    f"KL={m['kl']:.4f} top1={m['top1']:.4f} top10={m['top10']:.4f} "
                    f"({time.time()-t0:.1f}s)",
                    file=sys.stderr,
                )
                del v
                mx.clear_cache()
        n_pos += args.context_length
        del baseline
        mx.clear_cache()

    print(f"\n=== Results ({n_pos} positions) ===")
    print(f"{'Config':<22} {'KL':>10} {'Top-1':>10} {'Top-10':>10}")
    print("-" * 56)
    for bits in args.bit_configs:
        for sv in (False, True):
            r = results[(bits, sv)]
            kl = r["kl"] / n_pos
            t1 = r["top1"] / n_pos
            t10 = r["top10"] / n_pos
            label = f"bits={bits} sparse_v={'on' if sv else 'off'}"
            print(f"{label:<22} {kl:>10.4f} {t1:>10.4f} {t10:>10.4f}")
    print()
    print("Sparse V should produce essentially identical results.")
    print("Any meaningful difference indicates a bug.")


if __name__ == "__main__":
    main()
