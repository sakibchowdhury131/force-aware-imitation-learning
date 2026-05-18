#!/bin/bash
# Run any pipeline script in the ti conda environment.
# Usage: ./run.sh python 01_record.py [args...]
source /home/sakib/miniconda3/etc/profile.d/conda.sh
conda activate ti

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PIPELINE_DIR"

export PATH=/home/sakib/miniconda3/envs/ti/bin:$PATH
export CUDA_HOME=/home/sakib/miniconda3/envs/ti
export CUDAHOSTCXX=/home/sakib/miniconda3/envs/ti/bin/x86_64-conda-linux-gnu-g++
export CXX=/home/sakib/miniconda3/envs/ti/bin/x86_64-conda-linux-gnu-g++
export CC=/home/sakib/miniconda3/envs/ti/bin/x86_64-conda-linux-gnu-gcc
export TORCH_CUDA_ARCH_LIST="8.6"

"$@"
