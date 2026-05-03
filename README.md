# AWQ Benchmarking

Comparing AWQ, GPTQ, and RTN quantization on LLaMA-3-8B.

## Setup

Setup and reproduction instructions.

### 1. Hardware requirements

- NVIDIA GPU with CUDA Compute Capability 8.9+ (RTX 40-series / Ada Lovelace), 12 GB VRAM minimum
- 32 GB system RAM minimum
- 60 GB free disk space minimum
- Ubuntu 24.04 LTS (native or WSL2 on Windows 11)

### 2. WSL2 memory cap (skip if not using WSL2)

Create or edit `C:\Users\YOURNAME\.wslconfig`:

```
[wsl2]
memory=24GB
swap=16GB
```

From PowerShell:

```powershell
wsl --shutdown
```

### 3. System packages

```bash
sudo apt update
sudo apt install -y build-essential libpcre2-dev libpcre3-dev wget git
```

### 4. CUDA toolkits

System CUDA 12.0 toolkit is assumed already installed. Add CUDA 13.0:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-0
```

### 5. Miniconda

Skip if conda is already installed.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Restart your shell after install.

### 6. HuggingFace access

1. Create an account at https://huggingface.co
2. Request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B
3. Wait for approval (typically same-day)
4. Generate a read token at https://huggingface.co/settings/tokens

You will paste the token in step 8.

### 7. Clone this repository

```bash
cd ~
git clone <THIS_REPO_URL> awq-bench
cd awq-bench
```

### 8. Authenticate to HuggingFace

```bash
pip install --upgrade "huggingface_hub[cli]"
hf auth login
```

Paste your token when prompted. Decline git credential helper.

### 9. Clone llm-awq

```bash
cd ~
git clone https://github.com/mit-han-lab/llm-awq.git
```

### 10. Create the quantization environment

```bash
conda create -n awq-quantize python=3.10 -y
conda activate awq-quantize
pip install --upgrade pip
pip install -r ~/awq-bench/requirements-quantize.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

Build llm-awq into this environment:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd ~/llm-awq
pip install -e . --no-deps --no-build-isolation

cd ~/llm-awq/awq/kernels
python setup.py install
```

### 11. Create the evaluation environment

```bash
conda deactivate
conda create -n awq-eval python=3.10 -y
conda activate awq-eval
pip install --upgrade pip
pip install -r ~/awq-bench/requirements-eval.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

Build llm-awq into this environment as well (needed for AWQ evaluation):

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd ~/llm-awq
pip install -e . --no-deps --no-build-isolation

cd ~/llm-awq/awq/kernels
python setup.py install
```

### 12. Per-shell environment variables for AWQ

Any new shell that runs AWQ scripts (quantization or evaluation) must export these before running:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH
```

### 13. Find the LLaMA-3 snapshot path

After step 8, the model is downloaded on first use. Find its local path:

```bash
ls ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/
```

The output is a hash like `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`. Note this path; it is referenced as `LLAMA3_PATH` below. Full path:

```
/home/YOURNAME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/<HASH>
```

If the model has not been downloaded yet, run any of the quantization scripts in step 14 once to trigger the download (~16 GB).

### 14. Run quantization

```bash
conda activate awq-quantize
cd ~/awq-bench/scripts

python quantize_rtn.py
python quantize_gptq.py
```

For AWQ (set the env vars from step 12 first):

```bash
cd ~/llm-awq

python -m awq.entry \
    --model_path LLAMA3_PATH \
    --w_bit 4 --q_group_size 128 \
    --run_awq \
    --dump_awq ~/awq-bench/quantized/awq_cache/llama3-8b-w4-g128.pt

python -m awq.entry \
    --model_path LLAMA3_PATH \
    --w_bit 4 --q_group_size 128 \
    --load_awq ~/awq-bench/quantized/awq_cache/llama3-8b-w4-g128.pt \
    --q_backend real \
    --dump_quant ~/awq-bench/quantized/llama3-8b-awq-w4-g128/awq-model-w4-g128.pt
```

Replace `LLAMA3_PATH` with the path from step 13. Note: the second AWQ command auto-renames the output file with a `-v2` suffix.

### 15. Run evaluation

```bash
conda deactivate
conda activate awq-eval
cd ~/awq-bench/scripts
```

Set the env vars from step 12 if running AWQ.

```bash
python eval_perplexity.py --method fp16
python eval_perplexity.py --method rtn
python eval_perplexity.py --method gptq
python eval_perplexity.py --method awq

python eval_efficiency.py --method fp16
python eval_efficiency.py --method rtn
python eval_efficiency.py --method gptq
python eval_efficiency.py --method awq
```

Results land in `~/awq-bench/results/perplexity.csv` and `~/awq-bench/results/efficiency.csv`.

## Results

### Perplexity (lower is better)

| Method | WikiText-2 | C4 |
|--------|-----------:|------:|
| FP16   | 6.1358     | 8.9858 |
| RTN    | 6.6739     | 9.8866 |
| GPTQ   | 6.6065     | 10.0996 |
| AWQ    | 6.5312     | 9.6866 |

WikiText-2: full test split (288,627 tokens). C4: 256-sample slice of validation split (112,585 tokens). Both evaluated with non-overlapping 2048-token windows.

### Efficiency

| Method | Steady GPU Mem (MB) | Prefill (tok/s) | Generation (tok/s) | Generation Latency (s) |
|--------|--------------------:|----------------:|-------------------:|-----------------------:|
| FP16   | 9,339               | 1,339.2         | 1.1                | 243.17                 |
| GPTQ   | 5,733               | 3,663.8         | 10.7               | 23.89                  |
| AWQ    | 5,810               | 5,197.0         | 17.4               | 14.72                  |

Prefill: 2048-token forward pass. Generation: 256-token greedy decode from a fixed prompt. Each measurement averaged over 3 runs after 1 warmup. RTN omitted from efficiency benchmarks (stored as fake-quantized FP16; efficiency identical to FP16 baseline). RTN was not included because we didn't implement a "true" RTN algorithm; we simulated it for the purposes of perplexity analysis.