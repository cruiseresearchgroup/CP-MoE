import os
import argparse
import json
from rouge_score import rouge_scorer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', type=str, required=True, help="你的 SuperNI test.json 路径")
    parser.add_argument('--result_file', type=str, required=True, help="模型生成的预测结果 jsonl 路径")
    parser.add_argument('--output_dir', type=str, help="保存结果的路径")
    return parser.parse_args()

def eval_single(annotation_file, result_file, output_file=None):
    # ================= 1. 强健读取 Ground Truth =================
    loaded_data = []
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
    except json.decoder.JSONDecodeError:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    loaded_data.append(json.loads(line))
        
    if isinstance(loaded_data, dict) and "Instances" in loaded_data:
        annotations = loaded_data["Instances"]
    elif isinstance(loaded_data, list):
        annotations = loaded_data
    else:
        raise ValueError("无法解析的 annotation_file 格式")

    # ================= 2. 读取模型预测结果 =================
    with open(result_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]

    total = min(len(results), len(annotations))
    if total == 0:
        print("⚠️ 警告：有效样本数为 0，请检查输入文件。")
        return

    # 初始化 ROUGE 打分器 (对齐官方做法)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_rouge_l = 0.0

    # ================= 3. 计算 ROUGE-L =================
    for i in range(total):
        pred_text = results[i].get('text', '').strip().rstrip('.')        
        # 提取目标答案
        ground_truth = annotations[i].get('output', annotations[i].get('targets', []))
        if isinstance(ground_truth, str):
            valid_answers = [ground_truth]
        else:
            valid_answers = ground_truth

        # 遍历所有参考答案，取 ROUGE-L 最高分
        max_rouge_l = 0.0
        for ans in valid_answers:
            score = scorer.score(target=ans, prediction=pred_text)
            f1_score = score['rougeL'].fmeasure
            if f1_score > max_rouge_l:
                max_rouge_l = f1_score
                
        total_rouge_l += max_rouge_l

    # ================= 4. 输出结果 =================
    avg_rouge_l = 100. * total_rouge_l / total

    result_str = (
        f"Task: {os.path.basename(annotation_file)}\n"
        f"Samples: {total}\n"
        f"ROUGE-L Score: {avg_rouge_l:.2f}\n"
        f"----------------------------------------\n"
    )
    
    print(result_str)

    if output_file is not None:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(result_str)

if __name__ == "__main__":
    args = get_args()   
    eval_single(args.annotation_file, args.result_file, args.output_dir)