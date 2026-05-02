"""
Compute perplexity on WikiText-2 and C4 for FP16 baseline + 3 quantized models.

Usage:
    python eval_perplexity.py --method fp16          
    python eval_perplexity.py --method rtn
    python eval_perplexity.py --method gptq
    python eval_perplexity.py --method awq           

Results append to results/perplexity.csv.
"""

import argparse
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset

SEED = 42
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "perplexity.csv"

WINDOW_SIZE = 2048
DEVICE = "cuda"


def load_wikitext2_test():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    return text


def load_c4_validation_sample(num_samples=256):
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        samples.append(item["text"])
    text = "\n\n".join(samples)
    return text


@torch.no_grad()
def compute_perplexity(model, tokenizer, text, window=WINDOW_SIZE, label="dataset", max_tokens=None):
    """Sliding-window perplexity with non-overlapping stride."""
    print(f"  Tokenizing {label}...")
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encodings.input_ids[0]
    total_tokens = len(input_ids)

    # TODO
    if max_tokens is not None and max_tokens < total_tokens:
        input_ids = input_ids[:max_tokens]
        total_tokens = max_tokens
        print(f"  Truncated to {total_tokens:,} tokens for smoke test")

    print(f"  Total tokens: {total_tokens:,}")

    nll_sum = 0.0
    token_count = 0
    num_windows = total_tokens // window
    print(f"  Running {num_windows} forward passes (window={window})...")

    start_time = time.time()
    for i in range(num_windows):
        chunk = input_ids[i * window : (i + 1) * window].unsqueeze(0).to(DEVICE)
        outputs = model(chunk)
        # Compute loss manually for clarity over what's being averaged.
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = chunk[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nll_sum += loss.item()
        token_count += shift_labels.numel()

        if (i + 1) % 10 == 0 or i == num_windows - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (num_windows - i - 1) / rate if rate > 0 else 0
            print(f"  Window {i+1}/{num_windows}  "
                  f"running PPL={math.exp(nll_sum / token_count):.3f}  "
                  f"({rate:.1f} win/s, ETA {eta:.0f}s)")

    avg_nll = nll_sum / token_count
    ppl = math.exp(avg_nll)
    return ppl, token_count


def load_model_fp16():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "meta-llama/Meta-Llama-3-8B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def load_model_rtn():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = PROJECT_ROOT / "quantized" / "llama3-8b-rtn-w4-g128"
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def load_model_gptq():
    from gptqmodel import GPTQModel
    path = PROJECT_ROOT / "quantized" / "llama3-8b-gptq-w4-g128"
    model = GPTQModel.load(str(path))
    return model, model.tokenizer


def load_model_awq():
    raise NotImplementedError(
        "TODO"
    )


LOADERS = {
    "fp16": load_model_fp16,
    "rtn": load_model_rtn,
    "gptq": load_model_gptq,
    "awq": load_model_awq,
}


def append_result(method, dataset, ppl, num_tokens, runtime_sec):
    """Append a row to results CSV. Creates header if file doesn't exist."""
    header = "method,dataset,perplexity,num_tokens,runtime_sec\n"
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a") as f:
        if write_header:
            f.write(header)
        f.write(f"{method},{dataset},{ppl:.4f},{num_tokens},{runtime_sec:.1f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(LOADERS.keys()))
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"])
    parser.add_argument("--c4-samples", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=None, help="Truncate input for smoke testing")
    args = parser.parse_args()

    print(f"=== Method: {args.method} ===")
    print(f"Loading model...")
    model, tok = LOADERS[args.method]()
    print(f"Model loaded.")

    if "wikitext2" in args.datasets:
        print("\n--- WikiText-2 (test) ---")
        text = load_wikitext2_test()
        t0 = time.time()
        ppl, ntok = compute_perplexity(model, tok, text, label="wikitext2", max_tokens=args.max_tokens)
        elapsed = time.time() - t0
        print(f"WikiText-2 PPL: {ppl:.4f}  ({ntok:,} tokens, {elapsed:.1f}s)")
        append_result(args.method, "wikitext2", ppl, ntok, elapsed)

    if "c4" in args.datasets:
        print(f"\n--- C4 (validation, {args.c4_samples} samples) ---")
        text = load_c4_validation_sample(args.c4_samples)
        t0 = time.time()
        ppl, ntok = compute_perplexity(model, tok, text, label="c4", max_tokens=args.max_tokens)
        elapsed = time.time() - t0
        print(f"C4 PPL: {ppl:.4f}  ({ntok:,} tokens, {elapsed:.1f}s)")
        append_result(args.method, "c4", ppl, ntok, elapsed)

    print(f"\nResults appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()