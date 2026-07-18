# CP-MoE

Consistency-Preserving Mixture-of-Experts (LoRA-MoE) for continual instruction tuning, built on LLaVA / CL-MoE.

## Install

```bash
conda create -n cpmoe python=3.10 -y
conda activate cpmoe
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data

SuperNI tasks, one folder per task:

```
SuperNI/<task_name>/train.json
SuperNI/<task_name>/test.json
```

## Training

Each task has its own script under `scripts/LoraMoE/Train_NI/`. Train tasks sequentially
in order (the output of one task is the init for the next):

```bash
# single task
bash scripts/LoraMoE/Train_NI/1_task1572.sh

# full continual-learning order (task 1 → 15)
bash scripts/LoraMoE/Train_NI/Train.sh
```

Key arguments (set inside each script):

| Arg | Meaning |
| --- | --- |
| `--expert_num` | number of LoRA experts (default 8) |
| `--lora_r` / `--lora_alpha` | LoRA rank / alpha |
| `--cka_beta` | router CKA bias weight |
| `--use_cka_mask` | weight expert-mask update by CKA similarity (default `True`; set `False` to ablate) |
| `--warmup_tokens` | task-embedding warmup token budget |
| `--lora_target_modules` | modules to inject LoRA into |

Checkpoints are written to `--output_dir` (e.g. `./checkpoints/CL4VQA/<task>/...`).

## Evaluation

Each task has a matching eval script under `scripts/LoraMoE/Eval_NI/`.
Usage: `bash <script> <stage> <model_path> <cka_beta>`.

```bash
# single task
bash scripts/LoraMoE/Eval_NI/1_task1572.sh Finetune ./checkpoints/CL4VQA/task1572/llama-2-7b-hf-lora 0.2

# evaluate all tasks
bash scripts/LoraMoE/Eval_NI/Eval_all.sh
```

Each script runs multi-GPU inference (uses `CUDA_VISIBLE_DEVICES`), merges the
shards, and scores with `llava/eval/CLMoE/eval_superni.py`. Results and the
`accuracy_result.txt` land in `./results/CLMoE/<task>/<stage>/`.

## Acknowledgement

Built on [LLaVA](https://github.com/haotian-liu/LLaVA) and [CL-MoE](https://github.com/ECNU-ICALK/CL-MoE).
