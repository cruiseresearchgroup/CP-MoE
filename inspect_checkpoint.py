import torch
import os

# 换成你 task1 保存权重的路径
task1_path = "/srv/scratch/cruise/Yang/CL-MoE/checkpoints/CL4VQA/recognition-4ep-TS/llava-1.5-7b-lora/adapter_model.bin" 

if os.path.exists(task1_path):
    print("正在检查硬盘上的文件...")
    state_dict = torch.load(task1_path, map_location="cpu")
    
    # 随便找一个 B 矩阵看看
    b_keys = [k for k in state_dict.keys() if "lora_B" in k]
    if b_keys:
        sample_key = b_keys[0]
        sample_weight = state_dict[sample_key]
        print(f"Key: {sample_key}")
        print(f"Abs Sum: {sample_weight.abs().sum().item()}")
        
        if sample_weight.abs().sum().item() > 0:
            print("✅ 破案了！硬盘里的权重是有值的！是加载的时候弄丢了！")
        else:
            print("❌ 见鬼了，硬盘里的权重真的是 0 (那说明 Task 1 没训练)")
    else:
        print("没找到 LoRA B 的 key，可能保存格式不对")