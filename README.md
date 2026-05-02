# AWQ Benchmarking

Comparing AWQ, GPTQ, and RTN quantization on LLaMA-3-8B.

## Setup

**TODO**

## Methods

- **RTN**:
- **GPTQ**:
- **AWQ**:

## Results

(coming)


### Commands to remember:
**Verify CUDA 13 is installed**
ls /usr/local/cuda-13.0/bin/nvcc

**Point the build at CUDA 13 (these env vars only affect this terminal session)**
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

**Confirm nvcc is now 13.0**
nvcc --version

**Build the kernels**
conda activate awq
cd ~/llm-awq/awq/kernels
python setup.py install 2>&1 | tee /tmp/awq_kernel_build.log

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH