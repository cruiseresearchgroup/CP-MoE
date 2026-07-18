#!/bin/bash

MODELPATH="./checkpoints/CL4VQA/task875/llama-2-7b-hf-lora"
CKA_BETA="0.2"


bash ./scripts/LoraMoE/Eval_NI/1_task1572.sh  longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/2_task363.sh   longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/3_task1290.sh  longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/4_task181.sh   longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/5_task002.sh   longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/6_task1510.sh  longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/7_task639.sh   longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/8_task1729.sh  longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/9_task073.sh   longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/10_task1590.sh longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/11_task748.sh  longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/12_task511.sh  longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/13_task591.sh  longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/14_task1687.sh longseq $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI/15_task875.sh  longseq $MODELPATH $CKA_BETA
    