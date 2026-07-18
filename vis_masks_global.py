import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np
import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 这里填你所有任务的 output 目录路径 (按顺序排列)
# TASK_PATHS = [
#     # "./checkpoints/CL4VQA/task1572/llama-2-7b-hf-si",
#     "./checkpoints/CL4VQA/task363/llama-2-7b-hf-si",
#     # "./checkpoints/CL4VQA/location-4ep-entrophy/llava-1.5-7b-lora",
#     # "./checkpoints/CL4VQA/judge-4ep-entrophy/llava-1.5-7b-lora",
#     # "./checkpoints/CL4VQA/commonsense-4ep-entrophy/llava-1.5-7b-lora",
#     # "./checkpoints/CL4VQA/count-4ep-entrophy/llava-1.5-7b-lora",
#     # "./checkpoints/CL4VQA/action-4ep-entrophy/llava-1.5-7b-lora", 
#     # "./checkpoints/CL4VQA/color-4ep-TS/llava-1.5-7b-lora",
#     # "./checkpoints/CL4VQA/type-4ep-TS/llava-1.5-7b-lora",
#     # "./checkpoints/CL4VQA/subcategory-4ep-TS/llava-1.5-7b-lora",
#     # "./checkpoints/CL4VQA/causal-4ep-TS/llava-1.5-7b-lora",

# ]

TASK_PATHS = [
    # "./checkpoints/CL4VQA/task1572/llama-2-7b-hf-si",
    "./checkpoints/CL4VQA/task363/llama-2-7b-hf-si",
    # "./checkpoints/CL4VQA/location-4ep-entrophy/llava-1.5-7b-lora",
    # "./checkpoints/CL4VQA/judge-4ep-entrophy/llava-1.5-7b-lora",
    # "./checkpoints/CL4VQA/commonsense-4ep-entrophy/llava-1.5-7b-lora",
    # "./checkpoints/CL4VQA/count-4ep-entrophy/llava-1.5-7b-lora",
    # "./checkpoints/CL4VQA/action-4ep-entrophy/llava-1.5-7b-lora", 
    # "./checkpoints/CL4VQA/color-4ep-TS/llava-1.5-7b-lora",
    # "./checkpoints/CL4VQA/type-4ep-TS/llava-1.5-7b-lora",
    # "./checkpoints/CL4VQA/subcategory-4ep-TS/llava-1.5-7b-lora",
    # "./checkpoints/CL4VQA/causal-4ep-TS/llava-1.5-7b-lora",

]


# 2. 想要可视化的层关键字 (设为 None 则画所有层)
# 建议先只画 MLP 的 down_proj 和 up_proj 看看效果
FILTER_KEYWORDS = ["gate_proj", "up_proj", "down_proj"] 

# 3. 保存结果的目录
SAVE_ROOT = "./vis_global"
# ===========================================

def load_mask_file(path):
    """安全加载 mask 文件"""
    file_path = os.path.join(path, "clmoe_masks.pt")
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found: {file_path}")
        return None
    try:
        # map_location='cpu' 防止爆显存
        return torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def get_param_key(layer_name, param_name):
    """生成唯一的参数标识符"""
    return f"{layer_name}::{param_name}"

def main():
    print(f"🚀 开始全局 Mask 可视化分析...")
    print(f"📂 待分析任务数: {len(TASK_PATHS)}")
    
    # -------------------------------------------------
    # 第一步：全局扫描 (Find Global Max)
    # -------------------------------------------------
    print("\n[Step 1/2] Scanning for Global Max values...")
    
    # 存储每个参数的全局最大值
    # Key: "layer_name::param_name", Value: float (max value)
    global_max_registry = {} 
    
    # 存储这一层的参数是否是 2D 的 (只画矩阵)
    valid_2d_params = set()

    for task_dir in tqdm(TASK_PATHS, desc="Scanning Tasks"):
        state_dict = load_mask_file(task_dir)
        if state_dict is None: continue
        
        for layer_name, layer_data in state_dict.items():
            # 过滤关键字
            if FILTER_KEYWORDS and not any(k in layer_name for k in FILTER_KEYWORDS):
                continue
            
            expert_masks = layer_data.get('expert_masks', [])
            if not expert_masks: continue
            
            for expert_dict in expert_masks:
                if not expert_dict: continue
                
                for param_name, tensor in expert_dict.items():
                    # 只处理 2D 权重 (weight), 忽略 bias
                    if tensor.dim() != 2:
                        continue
                    
                    full_key = get_param_key(layer_name, param_name)
                    valid_2d_params.add(full_key)
                    
                    # 获取当前 tensor 的最大值
                    curr_max = tensor.float().max().item()
                    
                    # 更新全局记录
                    if full_key not in global_max_registry:
                        global_max_registry[full_key] = curr_max
                    else:
                        if curr_max > global_max_registry[full_key]:
                            global_max_registry[full_key] = curr_max

    print(f"✅ Global Max 统计完成！共找到 {len(global_max_registry)} 个参数矩阵。")
    
    # -------------------------------------------------
    # 第二步：统一绘图 (Plot with Fixed Scale)
    # -------------------------------------------------
    print("\n[Step 2/2] Plotting visualizations...")
    
    for task_idx, task_dir in enumerate(TASK_PATHS):
        task_name = os.path.basename(task_dir) # e.g., "task1_output"
        print(f"Processing Task {task_idx+1}/{len(TASK_PATHS)}: {task_name}")
        
        state_dict = load_mask_file(task_dir)
        if state_dict is None: continue
        
        for layer_name, layer_data in tqdm(state_dict.items(), desc=f"  Painting Layers", leave=False):
             # 过滤
            if FILTER_KEYWORDS and not any(k in layer_name for k in FILTER_KEYWORDS):
                continue

            expert_masks = layer_data.get('expert_masks', [])
            num_experts = len(expert_masks)
            if num_experts == 0: continue

            # 为了避免重复遍历，先收集这一层里所有的 param_names
            # 注意：必须只画刚才在 valid_2d_params 里注册过的
            current_layer_params = set()
            for em in expert_masks:
                if em: current_layer_params.update(em.keys())
            
            for param_name in current_layer_params:
                full_key = get_param_key(layer_name, param_name)
                
                if full_key not in valid_2d_params:
                    continue
                
                # === 核心：获取刚才算出来的全局最大值 ===
                # 哪怕这个 Task 里全是 0，也要用全局最大的那个值做上限
                # 这样才能看出它“很浅”
                g_max = global_max_registry[full_key]
                if g_max == 0: g_max = 1.0 # 防止全0报错

                # === 开始画图逻辑 (Grid Layout) ===
                cols = int(math.ceil(math.sqrt(num_experts)))
                rows = int(math.ceil(num_experts / cols))
                
                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
                if num_experts > 1: axes = axes.flatten()
                else: axes = [axes]
                
                fig.suptitle(f"Task: {task_name}\nLayer: {layer_name} | Param: {param_name}\nGlobal Max Scale: {g_max:.2e}", fontsize=12)
                
                has_valid_data = False
                for i in range(num_experts):
                    ax = axes[i]
                    mask_dict = expert_masks[i]
                    
                    if mask_dict and param_name in mask_dict:
                        data = mask_dict[param_name].detach().float().numpy()
                        sparsity = (data == 0).mean()
                        
                        # 🔥【关键】在这里应用全局统一的 vmax 🔥
                        sns.heatmap(data, 
                                    vmin=0, 
                                    vmax=g_max,   # <--- 所有的 Task 都在这里被统一了
                                    cmap="viridis", 
                                    cbar=True, 
                                    ax=ax,
                                    xticklabels=False, 
                                    yticklabels=False)
                        
                        ax.set_title(f"Expert {i}\n(Sparsity: {sparsity:.1%})", fontsize=10)
                        has_valid_data = True
                    else:
                        ax.text(0.5, 0.5, "No Mask", ha='center', color='gray')
                        ax.axis('off')

                # 隐藏多余子图
                for i in range(num_experts, len(axes)):
                    axes[i].axis('off')
                
                if has_valid_data:
                    # 保存路径结构: output/layer_name/param_name/task_X.png
                    # 这样你可以方便地在一个文件夹里通过切换图片看 Task 演变
                    safe_layer = layer_name.replace(".", "_")
                    safe_param = param_name.replace(".", "_")
                    
                    save_dir = os.path.join(SAVE_ROOT, safe_layer, safe_param)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    save_path = os.path.join(save_dir, f"step_{task_idx:02d}_{task_name}.png")
                    
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.85)
                    plt.savefig(save_path, dpi=100)
                    plt.close(fig)
                else:
                    plt.close(fig)

    print(f"\n✨ All Done! Results saved to: {SAVE_ROOT}")

if __name__ == "__main__":
    main()