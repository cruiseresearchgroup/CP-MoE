import json
import collections
import re
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_and_plot_routing_logs(input_folder, file_pattern='routing*log*.json'):
    """
    累加指定文件夹下所有日志的命中数据，生成一张汇总的热力图。
    """
    
    # 1. 搜索文件
    search_path = os.path.join(input_folder, file_pattern)
    log_files = glob.glob(search_path)
    
    if not log_files:
        print(f"❌ 未找到文件: {search_path}")
        return

    print(f"📊 正在汇总 {len(log_files)} 个文件的数据...")
    
    # 2. 初始化聚合容器
    # 使用 defaultdict(Counter) 来存储每一层的总命中数
    # 结构: { layer_idx: {expert_0: count, expert_1: count...} }
    aggregated_counts = collections.defaultdict(collections.Counter)
    
    # 3. 遍历文件并累加
    for file_path in log_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 筛选包含 'gate_proj' 的键
            gate_keys = [k for k in data['routing'].keys() if 'gate_proj' in k]
            
            for key in gate_keys:
                # 提取层号
                match = re.search(r'layers\.(\d+)\.', key)
                if match:
                    layer_idx = int(match.group(1))
                    
                    # 提取所有 token 的命中索引
                    topk_data = data['routing'][key]['topk_idx']
                    all_indices = [idx for sublist in topk_data for idx in sublist]
                    
                    # 【核心步骤】将当前文件的计数累加到总计数中
                    aggregated_counts[layer_idx].update(all_indices)
                    
        except Exception as e:
            print(f"⚠️ 跳过损坏的文件 {file_path}: {e}")

    # 4. 计算总比率并构建表格
    # 假设有 4 个专家
    num_experts = 8 
    rows = []
    
    # 对层号进行排序 (0, 1, 2 ... 31)
    sorted_layers = sorted(aggregated_counts.keys())
    
    for layer_idx in sorted_layers:
        counts = aggregated_counts[layer_idx]
        total_hits = sum(counts.values()) # 该层在所有文件中的总命中次数
        
        row = {'Layer': layer_idx}
        for exp in range(num_experts):
            if total_hits > 0:
                # 计算比率 = 该专家总命中 / 所有专家总命中
                row[f'Expert {exp}'] = counts[exp] / total_hits
            else:
                row[f'Expert {exp}'] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index('Layer', inplace=True)
    
    # 5. 保存数据和绘图
    output_base = os.path.join(input_folder, 'aggregated_summary')
    
    # 保存 CSV
    df.to_csv(f"{output_base}.csv")
    print(f"✅ 汇总 CSV 已保存: {output_base}.csv")

    # 绘制热力图
    plt.figure(figsize=(10, 12))
    sns.heatmap(df, annot=True, cmap='viridis', fmt=".2f", vmin=0, vmax=0.5)
    
    plt.title(f'Aggregated Expert Hit Ratios\n(Sum of {len(log_files)} Checkpoints)', fontsize=14)
    plt.ylabel('Layer Index')
    plt.xlabel('Expert Index')
    
    plt.tight_layout()
    plt.savefig(f"{output_base}_heatmap.png", dpi=150)


# ==========================================
# 👇 在这里修改文件夹路径
# ==========================================
if __name__ == "__main__":
    # 将 "." 替换为你的实际文件夹路径
    target_folder = "/srv/scratch/cruise/Yang/Lora-MoE/results/CLMoE/task1572/CL-CKA0.2/logs" 
    
    aggregate_and_plot_routing_logs(target_folder)