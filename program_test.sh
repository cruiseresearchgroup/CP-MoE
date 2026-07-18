#!/bin/bash
#PBS -N Lora_moe_warmup_Train_multi
#PBS -l select=1:ncpus=4:ngpus=2:gpu_model=H200:mem=120gb -l walltime=12:00:00
#PBS -j oe
#PBS -j oe

__conda_setup="$('/srv/scratch/z5539467/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/srv/scratch/z5539467/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/srv/scratch/z5539467/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/srv/scratch/z5539467/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_MMAP=1
export HF_HOME=/srv/scratch/z5539467/.cache/huggingface
export TRANSFORMERS_CACHE=/srv/scratch/z5539467/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1


## 把 torch 扩展与临时缓存全部挪到 scratch（防 quota）
#export TORCH_EXTENSIONS_DIR=/srv/scratch/CRUISE/Yang/torch_extensions
#export CUDA_CACHE_PATH=/srv/scratch/CRUISE/Yang/cuda_cache
#export TMPDIR=/srv/scratch/CRUISE/Yang/tmp
#export HUGGINGFACE_HUB_TOKEN=<YOUR_HF_TOKEN>
#export HF_HUB_DISABLE_MMAP=1
#mkdir -p "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"

# 小自检
echo ">>> Running GPU_test.py..."
conda activate loramoe
cd /srv/scratch/cruise/Yang/Lora-MoE/
bash scripts/LoraMoE/Train_NI/Train.sh
