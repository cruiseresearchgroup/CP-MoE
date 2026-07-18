import torch
import torch.nn as nn

from CLMoE.peft.tuners.TSmoe import CLMoEMOELoraLinear, CLMoEMOELoraConfig
def test_cl_moe_lifecycle():
    print("\n" + "="*60)
    print("开始 CL-MoE 全生命周期测试")
    print("="*60)

    # ---------------------------------------------------------
    # 1. 模拟配置
    # ---------------------------------------------------------
    batch_size = 4
    seq_len = 10
    in_features = 32
    out_features = 32
    adapter_name = "default"
    
    # 关键：设置极小的 warmup_tokens 以便快速触发阶段切换
    # 4 * 10 = 40 tokens per batch. 设置 100，大概 3 个 batch 就该触发了
    config_kwargs = {
        "r": 4,
        "lora_alpha": 8,
        "expert_num": 4,
        "task_embedding_dim": 8,
        "warmup_tokens": 1000, 
        "init_lora_weights": False
    }

    print(f"[1] 初始化模型层... (Warmup Threshold: {config_kwargs['warmup_tokens']} tokens)")
    layer = CLMoEMOELoraLinear(
        adapter_name=adapter_name,
        in_features=in_features,
        out_features=out_features,
        **config_kwargs
    ).cuda() # 如果有GPU

    # 模拟优化器
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    
    # ---------------------------------------------------------
    # 2. Phase 1: Warmup 阶段 (训练 TE)
    # ---------------------------------------------------------
    print("\n[2] 进入 Warmup 阶段 (Training TE)...")
    layer.train()
    
    for step in range(5):
        # 伪造输入数据
        x = torch.randn(batch_size, seq_len, in_features).cuda()
        
        # 前向传播
        output = layer(x)
        
        # 检查是否在使用 TE (通过 router 返回的 use_te 判断)
        # 这里我们hack一下直接看属性，或者看打印日志
        router = layer.lora_router[adapter_name]
        processed = router.processed_tokens.item()
        print(f"    Step {step}: Processed Tokens = {processed}")

        # 反向传播
        loss = output.mean()
        loss.backward()
        
        optimizer.step()
        
        # 【关键】必须手动调用积累梯度
        layer.accumulate_te_grad(adapter_name)
        optimizer.zero_grad()

        # ---------------------------------------------------------
        # 3. Phase 2: 触发分配 (Allocation)
        # ---------------------------------------------------------
        # 你的 forward 逻辑里写了：if warm_end: calculate_te_si()
        # 我们需要在外部模拟 allocate 的调用 (因为你把它放在了 forward 里注释掉了，或者外部调用)
        
        # 手动检查是否刚刚结束 warmup
        if router.warm_end:
            print(f"\n[3] Warmup 结束！触发 Allocation 逻辑...")
            
            # 模拟验证数据用于 CKA
            val_data = torch.randn(batch_size, seq_len, in_features).cuda()
            
            # 调用分配函数
            # 假设你按照我们之前的讨论修改了 forward，或者在这里手动调用
            # 注意：分配前先确保 SI 已经算好了 (forward 里算过了)
            
            target_expert = layer.allocate_expert_and_mask(
                adapter_name, 
                val_data, 
                cka_threshold=0.5, # 设置低一点方便触发 Case A/C，高一点触发 Case B
                si_threshold=0.01
            )
            
            # 验证分配结果
            print(f"    -> 分配目标专家: Expert {target_expert}")
            mask_content = layer.expert_masks[target_expert]
            if mask_content:
                print(f"    ✅ 检测到 Expert {target_expert} 的 Mask 已写入 (Keys: {len(mask_content)})")
            else:
                print(f"    ❌ 错误: Expert {target_expert} 的 Mask 依然为空！")
            
            # 此时 TE 应该被重置了
            break # 退出 Warmup 循环

    # ---------------------------------------------------------
    # 4. Phase 3: Stable 阶段 (测试正则化 Loss)
    # ---------------------------------------------------------
    print("\n[4] 进入 Stable 阶段 (Training SE with Regularization)...")
    
    # 必须先更新 Anchor (模拟任务切换/开始新阶段)
    layer.update_all_anchors()
    
    # 再跑几个 Step，看看参数变了之后，Loss 还会不会是 0
    x = torch.randn(batch_size, seq_len, in_features).cuda()
    
    # 让参数发生一点变化
    output = layer(x)
    (output.mean()).backward()
    optimizer.step() 
    
    # 计算正则化 Loss
    reg_loss = layer.get_regularization_loss()
    print(f"    当前正则化 Loss: {reg_loss.item()}")
    
    if reg_loss.item() > 0:
        print("    ✅ Regularization Loss 生效！(参数偏离了 Anchor，且 Mask 起作用了)")
    else:
        # 如果是第一次分配 (Case B)，Mask刚进去，Anchor刚更新，参数还没变，Loss 可能是 0
        # 尝试手动修改一下参数，强迫产生 Loss
        print("    (尝试手动扰动参数以验证 Loss...)")
        with torch.no_grad():
            layer.lora_A[adapter_name].loraA[target_expert].mlp.weight.add_(0.1)
        
        reg_loss_2 = layer.get_regularization_loss()
        print(f"    扰动后正则化 Loss: {reg_loss_2.item()}")
        if reg_loss_2.item() > 0:
            print("    ✅ Regularization Loss 验证成功。")
        else:
            print("    ❌ 警告: 参数已改变但 Loss 仍为 0，请检查 get_regularization_loss 逻辑。")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    test_cl_moe_lifecycle()