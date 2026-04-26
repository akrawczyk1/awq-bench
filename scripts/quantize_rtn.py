"""
RTN (round-to-nearest) 4-bit quantization baseline for LLaMA-3-8B.

This is "fake quantization": weights are quantized to int4 values then
dequantized back to FP16. This measures the accuracy impact of RTN without
needing custom int4 kernels. Memory and speed measurements from this baseline
are not meaningful — use GPTQModel/AWQ for those.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_tensor_rtn(weight: torch.Tensor, n_bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """
    Symmetric RTN quantization with grouping along the input dimension.

    Args:
        weight: 2D tensor of shape (out_features, in_features)
        n_bits: number of bits to quantize to (default 4)
        group_size: group size along in_features (default 128)

    Returns:
        Fake-quantized weight tensor, same shape and dtype as input.
    """
    assert weight.dim() == 2, f"Expected 2D weight, got shape {weight.shape}"
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, (
        f"in_features={in_features} not divisible by group_size={group_size}. "
        "Pad or pick a different group_size."
    )

    orig_dtype = weight.dtype
    # Compute in float32 for numerical stability, cast back at the end.
    w = weight.to(torch.float32)

    # Reshape to (out_features, num_groups, group_size).
    num_groups = in_features // group_size
    w_grouped = w.view(out_features, num_groups, group_size)

    # Symmetric quant range: [-q_max, q_max] where q_max = 2^(n_bits-1) - 1.
    # For 4-bit: q_max = 7, so we have 15 representable values: {-7, -6, ..., 6, 7}.
    q_max = 2 ** (n_bits - 1) - 1

    # Per-group scale: max absolute value in each group, divided by q_max.
    # Shape: (out_features, num_groups, 1) for broadcasting.
    abs_max = w_grouped.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / q_max
    # Avoid division by zero for any all-zero groups.
    scale = scale.clamp(min=1e-8)

    # Quantize: divide by scale, round, clamp to representable range.
    w_int = torch.round(w_grouped / scale).clamp(-q_max, q_max)

    # Dequantize: multiply back by scale. This is the "fake" part.
    w_dequant = w_int * scale

    # Reshape back and cast to original dtype.
    return w_dequant.view(out_features, in_features).to(orig_dtype)


def quantize_model_rtn(model: nn.Module, n_bits: int = 4, group_size: int = 128) -> dict:
    """
    Apply RTN quantization to all Linear layers in the model except the LM head.

    Modifies the model in place. Returns a dict with stats about the process.
    """
    # In LLaMA architectures, the final lm_head is not quantized. This matches
    # AWQ/GPTQ convention. Embedding layers also stay in FP16.
    skip_layers = {"lm_head"}

    stats = {"quantized_layers": 0, "skipped_layers": 0, "total_params_quantized": 0}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Skip the LM head and embedding-tied layers.
        if any(skip in name for skip in skip_layers):
            stats["skipped_layers"] += 1
            print(f"  [skip] {name}  shape={tuple(module.weight.shape)}")
            continue

        # in_features must be divisible by group_size. LLaMA-3-8B's hidden
        # sizes (4096, 14336, etc.) are all divisible by 128, so we're fine,
        # but assert defensively.
        if module.weight.shape[1] % group_size != 0:
            print(f"  [skip - bad shape] {name}  shape={tuple(module.weight.shape)}")
            stats["skipped_layers"] += 1
            continue

        with torch.no_grad():
            module.weight.data = quantize_tensor_rtn(
                module.weight.data, n_bits=n_bits, group_size=group_size
            )

        stats["quantized_layers"] += 1
        stats["total_params_quantized"] += module.weight.numel()
        print(f"  [quant] {name}  shape={tuple(module.weight.shape)}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B",
                        help="HF model ID or local path")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--output-dir", default="../quantized/llama3-8b-rtn-w4-g128")
    parser.add_argument("--test-prompt", default="The capital of France is")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} in FP16 to GPU...")
    print("(First run will download ~16GB. Subsequent runs use the HF cache.)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print(f"\nQuantizing all Linear layers to {args.bits}-bit, group_size={args.group_size}...")
    stats = quantize_model_rtn(model, n_bits=args.bits, group_size=args.group_size)
    print(f"\nDone. Quantized {stats['quantized_layers']} layers, "
          f"skipped {stats['skipped_layers']}, "
          f"total params quantized: {stats['total_params_quantized']:,}")

    # Quick sanity-check generation.
    print(f"\nSanity-check generation with prompt: {args.test_prompt!r}")
    inputs = tokenizer(args.test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    print("Generated:", tokenizer.decode(out[0], skip_special_tokens=True))

    # Save the fake-quantized model to disk for the eval phase.
    print(f"\nSaving fake-quantized model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Saved.")


if __name__ == "__main__":
    main()