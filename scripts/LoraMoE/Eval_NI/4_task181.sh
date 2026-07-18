#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


FILE_NAME="task181_outcome_extraction"
# MODEL_NAME="task181"
MODEL_NAME="task181"
TASK_NAME="task181"
if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH="./checkpoints/CL4VQA/${MODEL_NAME}/llama-2-7b-hf-lora"
else
    MODELPATH=$2
fi

if [ ! -n "$3" ] ;then
    echo "Missing required cka_beta argument. Usage: bash $0 <stage> <model_path> <cka_beta>"
    exit 1
else
    CKA_BETA=$3
fi

# 💡 核心改动 2：统一测试集路径 (确保你在切分时把后缀改成了 _test.json)
TEST_FILE="/srv/scratch/cruise/Yang/Lora-MoE/SuperNI/${FILE_NAME}/test.json"
RESULT_DIR="./results/CLMoE/${TASK_NAME}"

# 确保输出目录存在，防止报错
mkdir -p $RESULT_DIR/$STAGE

# ==================== 1. 多卡并行推理 ====================
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CLMoE.eval_token_monitor_NI \
        --model-path $MODELPATH \
        --model-base /srv/scratch/cruise/Yang/models/llama-2-7b-chat \
        --question-file $TEST_FILE \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --seed 42 \
        --cka_beta $CKA_BETA \
        --conv-mode llama-v2 & 
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# ==================== 2. 合并结果 ====================
# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# ==================== 3. 运行打分脚本 ====================
# 💡 核心改动 4：替换为你刚刚新建的纯文本 Eval 脚本，并且统一使用 TEST_FILE
python llava/eval/CLMoE/eval_superni.py \
    --annotation_file $TEST_FILE \
    --result_file $output_file \
    --output_dir $RESULT_DIR/$STAGE/accuracy_result.txt