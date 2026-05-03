"""
GPTQ 4-bit quantization for LLaMA-3-8B using GPTQModel.

Uses 128 calibration samples from WikiText-2 train, matching the
AWQ paper's calibration setup for fair comparison.
"""

"""
Code written with the help of Claude Opus 4.7
Main usages:
- Making sure my implementation of the quantization process is correct and matches the expected API of GPTQModel.
- Implementing the terminal window progress outputs.
- Debugging the overall flow of loading, quantizing, and saving the model.
"""

import argparse
import random
from pathlib import Path

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


def get_calibration_dataset(num_samples: int = 128, min_length: int = 1000) -> list[str]:
    """
    Load WikiText-2 train, filter short/empty entries, and sample.

    GPTQModel internally further filters via calibration_dataset_min_length,
    but pre-filtering speeds things up and gives us reproducible sampling.
    """
    print(f"Loading WikiText-2 train split for calibration...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    samples = [s for s in ds["text"] if len(s.strip()) >= min_length]
    print(f"  {len(samples)} samples after filtering (min_length={min_length})")
    random.shuffle(samples)
    samples = samples[:num_samples]
    print(f"  Selected {len(samples)} samples for calibration.")
    return samples


def main():
    parser = argparse.ArgumentParser()
    # Auto-discover the local LLaMA-3 snapshot path.
    snapshots_dir = Path.home() / ".cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots"
    default_model = None
    if snapshots_dir.exists():
        snapshot_dirs = list(snapshots_dir.iterdir())
        if snapshot_dirs:
            default_model = str(snapshot_dirs[0])

    parser.add_argument(
        "--model",
        default=default_model,
        help="Path to local LLaMA-3-8B snapshot. Auto-discovered if not provided.",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-calib-samples", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "quantized" / "llama3-8b-gptq-w4-g128"),
    )
    parser.add_argument("--test-prompt", default="The capital of France is")
    args = parser.parse_args()
    if args.model is None:
        raise RuntimeError(
            "Could not auto-discover LLaMA-3-8B snapshot. "
            "Either run an HF download first, or pass --model with an explicit path."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calibration data
    calibration_dataset = get_calibration_dataset(num_samples=args.num_calib_samples)

    # Quantization config: matches RTN settings for fair comparison.
    # NOTE: act_group_aware (GAR) intentionally disabled to keep this as a
    # vanilla GPTQ baseline matching the AWQ paper.
    quant_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=True,
        desc_act=False,
    )

    print(f"\nLoading {args.model} with GPTQModel...")
    model = GPTQModel.load(args.model, quant_config)

    print(f"\nQuantizing with {args.num_calib_samples} calibration samples...")
    print("(Expect 10-30 minutes on a 4070 Super for an 8B model.)")
    model.quantize(calibration_dataset, batch_size=1)

    print(f"\nSaving quantized model to {output_dir}...")
    model.save(str(output_dir))
    print("Saved.")

    # Sanity check: Reload from disk and generate to verify the saved checkpoint is functional.
    print(f"\nReloading saved model for sanity check...")
    del model
    torch.cuda.empty_cache()
    model = GPTQModel.load(str(output_dir))
    result = model.generate(args.test_prompt, max_new_tokens=30)[0]
    print(f"Prompt: {args.test_prompt!r}")
    print(f"Generated: {model.tokenizer.decode(result)}")


if __name__ == "__main__":
    main()