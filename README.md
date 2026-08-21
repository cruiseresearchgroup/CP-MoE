# CP-MoE: Consistency-Preserving Mixture-of-Experts

CP-MoE is a continual-learning framework for the sequential fine-tuning of large foundation models with LoRA-based Mixture-of-Experts, designed to mitigate catastrophic forgetting.

When LLMs and vision-language models adapt to a stream of tasks with LoRA-MoE, two failure modes arise: **aggressive isolation** (restricting experts too strictly blocks reuse of previously learned knowledge on similar new tasks) and **aggressive merging** (overwriting existing parameters with new-task updates erases past tasks). CP-MoE resolves this trade-off by first probing each new task with a temporary expert, then using what it learns to guide how the stable experts are routed and updated.

## Method

- **Transient Expert** — a temporary expert module that captures early, task-specific updates during the initial phase of a new task and is discarded afterwards.
- **Consistency-Preserving Routing Bias** — the transient expert is used to measure representation similarity between the new data and the existing stable experts, steering routing toward the most compatible stable experts.
- **Transient Expert-Guided Regularisation** — when the stable experts are updated, the mechanism identifies which historical parameters are most critical and applies targeted regularisation to shield them from being overwritten.

## Results

Evaluated on frozen 7B backbones (Llama-2-7B for language, LLaVA-1.5-7B for vision-language), training only the LoRA experts and router (1.48% of parameters on SuperNI, 0.47% on VQA v2):

| Benchmark | Setting | CP-MoE | Strongest baseline |
| --- | --- | --- | --- |
| SuperNI | 8 sequential language tasks | **50.84%** avg / **1.32%** forgetting | 51.54% (GainLoRA) |
| SuperNI | zero-shot transfer, 7 unseen tasks | **35.80%** | 33.80% (GainLoRA) |
| VQA v2 | 10 sequential visual-reasoning tasks | **62.30%** avg / **0.35%** forgetting | 60.77% / 1.77% (CL-MoE) |

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

## License

Released under the [MIT License](LICENSE).

## Acknowledgement

Built on [LLaVA](https://github.com/haotian-liu/LLaVA) and [CL-MoE](https://github.com/ECNU-ICALK/CL-MoE).
