################## VICUNA ##################
# PROMPT_VERSION="plain" 
# MODEL_VERSION="Llama-2-7b-hf"
################## VICUNA ##################


################## LLaMA-2 ##################
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

deepspeed --include localhost:0,1 --master_port 29613 llava/train/train_mem_MOE.py \
    --deepspeed ./scripts/zero1_offload.json \
    --lora_enable True --lora_r 32 --lora_alpha 32    \
    --expert_num 8 \
    --model_name_or_path /srv/scratch/cruise/Yang/models/llama-2-7b-chat \
    --version $PROMPT_VERSION \
    --data_path /srv/scratch/cruise/Yang/Lora-MoE/SuperNI/task748_glucose_reverse_cause_event_detection/train.json \
    --image_folder /srv/scratch/cruise/Yang/ \
    --previous_task_model_path ./checkpoints/CL4VQA/task1590/llama-2-7b-hf-lora \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/CL4VQA/task748/llama-2-7b-hf-lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --save_total_limit  1 \
    --learning_rate 2e-04 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --lora_target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    --task quoref \
    --lora_dropout 0.1 \
    --warmup_tokens 10000 \
    --cka_beta 0.2 \