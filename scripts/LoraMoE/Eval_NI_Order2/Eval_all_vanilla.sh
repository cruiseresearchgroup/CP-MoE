#!/bin/bash

BASE_MODEL=/srv/scratch/cruise/Yang/models/llama-2-7b-chat

bash ./scripts/LoraMoE/Eval_NI_Order2/1_task748.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/2_task073.sh   Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/3_task1590.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/4_task639.sh   Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/5_task1572.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/6_task1687.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/7_task591.sh   Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/8_task363.sh   Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/9_task1510.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/10_task1729.sh Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/11_task181.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/12_task511.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/13_task002.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/14_task1290.sh Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI_Order2/15_task875.sh  Vanilla $BASE_MODEL
