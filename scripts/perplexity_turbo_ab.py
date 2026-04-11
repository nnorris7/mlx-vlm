#!/usr/bin/env python3
"""
Perplexity A/B test for TurboQuant K/V direction.

Computes perplexity on WikiText-2 using mlx-vlm's language model backbone,
with TurboQuant KV cache quantization applied via the same code path as
`mlx_vlm generate`. Supports the TURBO_K_GETS_MORE_BITS env var to flip the
asymmetric direction.

Usage:
    # Baseline (no quantization)
    .venv/bin/python scripts/perplexity_turbo_ab.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit

    # Variant A: V gets more bits (current default)
    .venv/bin/python scripts/perplexity_turbo_ab.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit --kv-bits 2.5

    # Variant B: K gets more bits (Turney's direction)
    TURBO_K_GETS_MORE_BITS=1 .venv/bin/python scripts/perplexity_turbo_ab.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit --kv-bits 2.5
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


def compute_window_nll(model, window_tokens, kv_bits, kv_quant_scheme):
    """Run a single forward pass over a window and return (total_nll, n_tokens_scored)."""
    input_ids = mx.array(window_tokens)[None, :]  # [1, L]
    L = input_ids.shape[1]

    # Build fresh cache for this window via the language model's make_cache
    lm = model.language_model if hasattr(model, "language_model") else model
    cache = lm.make_cache()

    # Apply TurboQuant quantization at token 0 (force, not deferred)
    if kv_bits is not None:
        maybe_quantize_kv_cache(
            cache,
            quantized_kv_start=0,
            kv_group_size=64,
            kv_bits=kv_bits,
            kv_quant_scheme=kv_quant_scheme,
        )

    # Forward pass via top-level model so PLE (per-layer inputs) are computed
    out = model(input_ids, cache=cache)
    logits = out.logits  # [1, L, V]

    # Shift: logits[:, :-1] predicts input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Compute log p(target) per position
    log_probs = nn.log_softmax(shift_logits.astype(mx.float32), axis=-1)
    target_log_probs = mx.take_along_axis(
        log_probs, shift_labels[:, :, None], axis=-1
    ).squeeze(-1)  # [1, L-1]

    total_nll = float(-target_log_probs.sum())
    n_tokens = shift_labels.size

    # Release memory between windows
    mx.eval(logits)
    del logits, shift_logits, log_probs, target_log_probs, out, cache
    mx.clear_cache()

    return total_nll, n_tokens


def compute_perplexity(model, token_stream, context_length, kv_bits, kv_quant_scheme, bos_id=None):
    """Compute perplexity by splitting the token stream into non-overlapping windows."""
    total = len(token_stream)
    total_nll = 0.0
    total_scored = 0
    n_windows = total // context_length

    if n_windows == 0:
        raise ValueError(
            f"Not enough tokens ({total}) for one window of {context_length}"
        )

    for i in range(n_windows):
        start = i * context_length
        end = start + context_length
        window = token_stream[start:end]
        # Prepend BOS so the model sees a clean context start
        if bos_id is not None:
            window = [bos_id] + list(window[:-1])

        t0 = time.time()
        window_nll, window_tokens = compute_window_nll(
            model, window, kv_bits, kv_quant_scheme
        )
        elapsed = time.time() - t0

        total_nll += window_nll
        total_scored += window_tokens

        window_ppl = math.exp(window_nll / window_tokens)
        running_ppl = math.exp(total_nll / total_scored)
        print(
            f"  window {i + 1}/{n_windows}: "
            f"window_ppl={window_ppl:.4f}  running_ppl={running_ppl:.4f}  "
            f"({elapsed:.1f}s)",
            file=sys.stderr,
        )

    return math.exp(total_nll / total_scored), total_scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="TurboQuant bits (None for baseline, e.g. 2.5, 3.5, 4)",
    )
    parser.add_argument("--kv-quant-scheme", default="turboquant")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens from the corpus (will be rounded to context_length)",
    )
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    # Build label
    k_more = os.environ.get("TURBO_K_GETS_MORE_BITS", "0") == "1"
    if args.kv_bits is None:
        label = "baseline"
    else:
        label = f"turbo{args.kv_bits}_{'K-more' if k_more else 'V-more'}"

    print(f"=== {label} ===", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Context length: {args.context_length}", file=sys.stderr)

    # Load model
    print("Loading model...", file=sys.stderr)
    t0 = time.time()
    model, processor = load(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    tokenizer = processor.tokenizer

    # Load dataset
    print(f"Loading {args.dataset}/{args.dataset_config} ({args.split})...", file=sys.stderr)
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    # Tokenize
    print("Tokenizing...", file=sys.stderr)
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    all_tokens = all_tokens[: args.max_tokens]
    # Prepend BOS so each window starts cleanly (Gemma 4 expects it)
    bos_id = tokenizer.bos_token_id
    print(f"  {len(all_tokens)} tokens  bos_id={bos_id}", file=sys.stderr)

    # Compute perplexity
    print("Computing perplexity...", file=sys.stderr)
    t0 = time.time()
    ppl, n_scored = compute_perplexity(
        model,
        all_tokens,
        context_length=args.context_length,
        kv_bits=args.kv_bits,
        kv_quant_scheme=args.kv_quant_scheme,
        bos_id=bos_id,
    )
    elapsed = time.time() - t0

    print(
        f"\n{label}: PPL={ppl:.4f} "
        f"({n_scored} tokens scored, {elapsed:.1f}s total)"
    )


if __name__ == "__main__":
    main()
