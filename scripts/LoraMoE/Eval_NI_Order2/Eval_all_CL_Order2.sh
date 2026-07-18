#!/bin/bash

MODELPATH="./checkpoints_order2/CL4VQA/task363/llama-2-7b-hf-lora"
CKA_BETA="0.2"

bash ./scripts/LoraMoE/Eval_NI_Order2/1_task748.sh   order2 $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/2_task073.sh   order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/3_task1590.sh  order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/4_task639.sh   order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/5_task1572.sh  order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/6_task1687.sh  order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/7_task591.sh   order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/8_task363.sh   order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/9_task1510.sh  order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/10_task1729.sh order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/11_task181.sh  order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/12_task511.sh  order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/13_task002.sh  order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/14_task1290.sh order2  $MODELPATH $CKA_BETA
bash ./scripts/LoraMoE/Eval_NI_Order2/15_task875.sh  order2  $MODELPATH $CKA_BETA
