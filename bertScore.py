import os
import json
import numpy as np
from collections import defaultdict
from bert_score import score

def batch_evaluate_bertscore_comparative(pred_base_dir, unified_ref_path, task_list):
    # 1. 解析统一参考文件，同时获取 Ground Truth 和 Baseline Predictions
    print(f"Loading unified reference file: {unified_ref_path}...")
    grouped_refs = defaultdict(list)
    
    if not os.path.exists(unified_ref_path):
        print(f"  [Error] Unified reference file not found: {unified_ref_path}")
        return

    with open(unified_ref_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            dataset_name = data.get("Dataset", "")
            task_name = dataset_name.split("_")[0]
            grouped_refs[task_name].append(data)

    print(f"Successfully loaded and grouped data for {len(grouped_refs)} tasks.\n")

    results_model1 = {}    # 存储你的模型 (merge.jsonl) 的结果
    results_baseline = {}  # 存储 Baseline 预测的结果
    all_f1_m1, all_f1_base = [], []

    # 2. 遍历任务
    for task in task_list:
        print(f"Processing {task}...")
        
        pred_path = os.path.join(pred_base_dir, task, "Finetune", "merge.jsonl")
        
        if not os.path.exists(pred_path):
            print(f"  [Error] Prediction file not found: {pred_path}")
            continue
            
        task_refs = grouped_refs.get(task, [])
        if not task_refs:
            print(f"  [Warning] No references found for {task}.")
            continue

        model1_preds = []
        baseline_preds = []
        references = []

        # 3. 逐行对齐提取三方数据
        with open(pred_path, "r", encoding="utf-8") as f_pred:
            pred_lines = f_pred.readlines()
            
            for pred_line, ref_data in zip(pred_lines, task_refs):
                pred_data = json.loads(pred_line.strip())
                
                # 你的模型输出
                m1_text = pred_data.get("text", "").strip()
                # Baseline 模型的输出
                base_text = ref_data.get("Prediction", "").strip()
                # 真实标签
                ref_text = ref_data.get("Instance", {}).get("ground_truth", "").strip()
                
                if m1_text and base_text and ref_text:
                    model1_preds.append(m1_text)
                    baseline_preds.append(base_text)
                    references.append(ref_text)

        if not references:
            print(f"  [Warning] No valid aligned data for {task}.")
            continue

        # 4. 分别计算两个模型的 BERTScore
        print(f"  -> Scoring Model 1 (merge.jsonl)...")
        P1, R1, F1_1 = score(model1_preds, references, lang="en", rescale_with_baseline=True, verbose=False)
        
        print(f"  -> Scoring Baseline...")
        P2, R2, F1_2 = score(baseline_preds, references, lang="en", rescale_with_baseline=True, verbose=False)
        
        # 记录 F1 均值
        f1_m1_mean = F1_1.mean().item()
        f1_base_mean = F1_2.mean().item()
        
        results_model1[task] = f1_m1_mean
        results_baseline[task] = f1_base_mean
        
        all_f1_m1.append(f1_m1_mean)
        all_f1_base.append(f1_base_mean)
        
        print(f"  [Done] Model 1: {f1_m1_mean:.4f} | Baseline: {f1_base_mean:.4f}\n")

    # 5. 打印对比总表 (直接可用作论文表格)
    print("\n" + "=" * 65)
    print(f"{'Task Name':<12} | {'Model 1 (merge) F1':<18} | {'Baseline F1':<15} | {'Delta':<8}")
    print("-" * 65)
    
    for task in results_model1:
        m1_f1 = results_model1[task]
        base_f1 = results_baseline[task]
        delta = m1_f1 - base_f1
        sign = "+" if delta > 0 else ""
        print(f"{task:<12} | {m1_f1:.4f}             | {base_f1:.4f}         | {sign}{delta:.4f}")
        
    print("-" * 65)
    if all_f1_m1:
        avg_m1 = np.mean(all_f1_m1)
        avg_base = np.mean(all_f1_base)
        delta_avg = avg_m1 - avg_base
        sign_avg = "+" if delta_avg > 0 else ""
        print(f"{'MACRO AVG':<12} | {avg_m1:.4f}             | {avg_base:.4f}         | {sign_avg}{delta_avg:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    PRED_BASE_DIR = "./results/CLMoE"
    UNIFIED_REF_PATH = "./predict_eval_predictions.jsonl" 
    
    TARGET_TASKS = [
        "task1572", "task363", "task1290", "task181", 
        "task002", "task1510", "task639", "task1729", 
        "task073", "task1590", "task748", "task511", 
        "task591", "task1687", "task875"
    ]
    
    batch_evaluate_bertscore_comparative(PRED_BASE_DIR, UNIFIED_REF_PATH, TARGET_TASKS)