#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/CL4VQA/recognition-8ep/llava-1.5-7b-lora'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/CLMoE/recognition"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CLMoE.eval_token_monitor \
        --model-path $MODELPATH \
        --model-base /srv/scratch/cruise/Yang/models/vicuna-7b-v1.5 \
        --question-file /srv/scratch/cruise/Yang/CL4VQA/test/sample.json \
        --image-folder /srv/scratch/cruise/Yang/ \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/CLMoE/eval_vqav2.py \
    --annotation_file /srv/scratch/cruise/Yang/CL4VQA/test/sample.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE/output_result.jsonl \
