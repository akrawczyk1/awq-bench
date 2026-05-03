"""
Efficiency benchmarks: peak memory, throughput, generation latency.

Measures four metrics per method:
  - Load-time peak GPU memory (transient)
  - Steady-state GPU memory after model load
  - Throughput (tokens/sec) on a 2048-token prefill, batch 1
  - End-to-end latency for greedy generation of 256 tokens

All measurements averaged over 3 runs after a warmup. Each method is run
in a separate process invocation to ensure clean GPU state.

Usage:
    python eval_efficiency.py --method fp16
    python eval_efficiency.py --method rtn
    python eval_efficiency.py --method gptq
    python eval_efficiency.py --method awq

Results append to results/efficiency.csv.
"""

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch

SEED = 42
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "efficiency.csv"

DEVICE = "cuda"
PREFILL_LENGTH = 2048
GENERATION_LENGTH = 256
NUM_RUNS = 3
WARMUP_RUNS = 1
PROMPT = "The capital of France is"


# === Loaders (copied from eval_perplexity.py for self-containedness) ===

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
    LLM_AWQ_PATH = str(Path.home() / "llm-awq")
    if LLM_AWQ_PATH not in sys.path:
        sys.path.insert(0, LLM_AWQ_PATH)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
    from awq.quantize.quantizer import real_quantize_model_weight
    from awq.utils.utils import simple_dispatch_model

    base_model_path = str(
    Path.home() / ".cache/huggingface/hub/"
    "models--meta-llama--Meta-Llama-3-8B/snapshots"
)
    # Resolve to the actual snapshot directory (there's exactly one)
    snapshot_dirs = list(Path(base_model_path).iterdir())
    if not snapshot_dirs:
        raise RuntimeError(f"No LLaMA-3 snapshot found in {base_model_path}. Run a quantization script first to trigger download.")
    base_model_path = str(snapshot_dirs[0])
    quant_path = str(
        PROJECT_ROOT / "quantized" / "llama3-8b-awq-w4-g128" / "awq-model-w4-g128-v2.pt"
    )

    q_config = {"zero_point": True, "q_group_size": 128}
    w_bit = 4

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.float16, trust_remote_code=True
        )

    real_quantize_model_weight(model, w_bit=w_bit, q_config=q_config, init_only=True)
    model.tie_weights()
    device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer"])
    load_checkpoint_in_model(
        model, checkpoint=quant_path, device_map=device_map, offload_state_dict=True,
    )
    model = simple_dispatch_model(model, device_map=device_map)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    return model, tokenizer


LOADERS = {
    "fp16": load_model_fp16,
    "rtn": load_model_rtn,
    "gptq": load_model_gptq,
    "awq": load_model_awq,
}


# === Measurement helpers ===

def gpu_mem_used_mb():
    """Currently-allocated GPU memory in MB."""
    return torch.cuda.memory_allocated() / 1e6


def reset_peak_memory():
    torch.cuda.reset_peak_memory_stats()


def gpu_peak_mem_mb():
    """Peak allocated GPU memory in MB since last reset."""
    return torch.cuda.max_memory_allocated() / 1e6


@torch.no_grad()
def measure_prefill_throughput(model, tokenizer):
    """Tokens/sec for a single 2048-token forward pass."""
    # Build a 2048-token input by repeating the prompt.
    prompt_tokens = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids[0]
    repeats = (PREFILL_LENGTH // len(prompt_tokens)) + 1
    input_ids = prompt_tokens.repeat(repeats)[:PREFILL_LENGTH].unsqueeze(0).to(DEVICE)

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = model(input_ids)
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(input_ids)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg_time = statistics.mean(times)
    tokens_per_sec = PREFILL_LENGTH / avg_time
    return tokens_per_sec, avg_time


@torch.no_grad()
def measure_generation_latency(model, tokenizer):
    """Wall-clock for greedy generation of 256 tokens from a fixed prompt.

    Forces exactly GENERATION_LENGTH tokens via min_new_tokens to avoid
    early-termination via EOS, which would inflate measured throughput.
    """
    inputs = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

    gen_kwargs = dict(
        max_new_tokens=GENERATION_LENGTH,
        min_new_tokens=GENERATION_LENGTH,
        do_sample=False,
    )

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = model.generate(inputs, **gen_kwargs)
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model.generate(inputs, **gen_kwargs)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg_time = statistics.mean(times)
    return avg_time, GENERATION_LENGTH / avg_time


def append_result(method, load_peak_mb, steady_mb, prefill_tps, gen_latency, gen_tps):
    header = "method,load_peak_mb,steady_mb,prefill_tokens_per_sec,gen_latency_sec,gen_tokens_per_sec\n"
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a") as f:
        if write_header:
            f.write(header)
        f.write(f"{method},{load_peak_mb:.0f},{steady_mb:.0f},"
                f"{prefill_tps:.1f},{gen_latency:.2f},{gen_tps:.1f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(LOADERS.keys()))
    args = parser.parse_args()

    print(f"=== Method: {args.method} ===")

    # Reset GPU state
    torch.cuda.empty_cache()
    gc.collect()
    reset_peak_memory()
    mem_before_load = gpu_mem_used_mb()
    print(f"GPU memory before load: {mem_before_load:.0f} MB")

    print(f"Loading model...")
    t0 = time.time()
    model, tok = LOADERS[args.method]()
    load_time = time.time() - t0
    load_peak = gpu_peak_mem_mb()
    steady_mem = gpu_mem_used_mb()
    print(f"Model loaded in {load_time:.1f}s.")
    print(f"  Load-time peak GPU mem: {load_peak:.0f} MB")
    print(f"  Steady-state GPU mem:   {steady_mem:.0f} MB")

    print(f"\nMeasuring prefill throughput ({PREFILL_LENGTH} tokens, "
          f"{NUM_RUNS} runs after {WARMUP_RUNS} warmup)...")
    prefill_tps, prefill_time = measure_prefill_throughput(model, tok)
    print(f"  Prefill: {prefill_tps:.1f} tokens/sec ({prefill_time*1000:.0f} ms)")

    print(f"\nMeasuring generation latency ({GENERATION_LENGTH} tokens, greedy, "
          f"{NUM_RUNS} runs after {WARMUP_RUNS} warmup)...")
    gen_latency, gen_tps = measure_generation_latency(model, tok)
    print(f"  Generation: {gen_latency:.2f}s ({gen_tps:.1f} tokens/sec)")

    append_result(args.method, load_peak, steady_mem, prefill_tps, gen_latency, gen_tps)
    print(f"\nResults appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()