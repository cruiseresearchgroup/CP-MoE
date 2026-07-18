#!/bin/bash

BASE_MODEL=/srv/scratch/cruise/Yang/models/llama-2-7b-chat

bash ./scripts/LoraMoE/Eval_NI/1_task1572.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/2_task363.sh   Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/3_task1290.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/4_task181.sh   Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/5_task002.sh   Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/6_task1510.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/7_task639.sh   Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/8_task1729.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/9_task073.sh   Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/10_task1590.sh Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/11_task748.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/12_task511.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/13_task591.sh  Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/14_task1687.sh Vanilla $BASE_MODEL
bash ./scripts/LoraMoE/Eval_NI/15_task875.sh  Vanilla $BASE_MODEL
