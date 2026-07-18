#!/bin/bash

MODELPATH="./checkpoints_order2/CL4VQA/task875/llama-2-7b-hf-lora"
CKA_BETA="0.2"

bash ./scripts/LoraMoE/Eval_NI_Order2/1_task748.sh  cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/2_task073.sh   cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/3_task1590.sh  cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/4_task639.sh   cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/5_task1572.sh  cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/6_task1687.sh  cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/7_task591.sh   cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/8_task363.sh   cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/9_task1510.sh  cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/10_task1729.sh cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/11_task181.sh  cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/12_task511.sh  cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/13_task002.sh  cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/14_task1290.sh cka0.2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/15_task875.sh  cka0.2 $MODELPATH $CKA_BETA
