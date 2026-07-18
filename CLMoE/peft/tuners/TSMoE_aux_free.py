# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
from dataclasses import dataclass, field
import copy

import numpy as np
# import tensorflow as tf
import deepspeed

import torch
import torch.nn as nn
import torch.nn.functional as F

from pprint import pprint
from deepspeed import zero

from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)


from .CKA_metric import linear_cka
from ..import_utils import is_bnb_4bit_available, is_bnb_available
if is_bnb_available():
    import bitsandbytes as bnb

# ==========================================
# 1. Config Definition
# ==========================================
@dataclass
class CLMoEMOELoraConfig(LoraConfig):
    #todo currently we setting the warmup_tokens to 100000, futher can be replaced by HT-SR metrics.
    warmup_tokens: int = field(default=10000)
    """
    This is the configuration class to store the configuration of a [`~peft.MOE_LORA_CLMoE`]
    """
    task_embedding_dim: int = field(default=64)
    # Fix: 使用 default_factory 防止可变参数陷阱
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"])
    expert_num: int = field(default=4)
    router_bias_update_rate: float = field(default=0.01)  # u for expert bias
    router_tolerance_ratio: float = field(default=0.2) # tao for tolerance
    cka_beta: float = field(default=0.001)  # alpha for CKA structural inductive bias (Eq.9)
    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CLMoE

# ==========================================
# 2. Components (Router, Experts, moved UP)
# ==========================================



class TSMoERouter(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.num_experts = config.expert_num 
        self.classifier = nn.Linear(
            in_features, 
            self.num_experts, 
            bias=getattr(config, "router_bias", False)
        )
        self.register_buffer("expert_similarities", torch.zeros(self.num_experts))        
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        self.beta = getattr(config, "cka_beta", 0.001)
        self.bias_update_rate = getattr(config, "router_bias_update_rate", 0.01)
        self.threshold = getattr(config, "warmup_tokens", 650)
        self.register_buffer("processed_tokens", torch.tensor(0, dtype=torch.long))
        self.register_buffer("warm_end", torch.tensor(False, dtype=torch.bool))
        self.jitter_noise = getattr(config, "router_jitter_noise", 0.0)
        torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
    
    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            计算路由结果：
            1. 使用原始 affinity 作为 expert 权重来源；
            2. 使用 affinity + bias 仅进行 top-k 专家选择；
            3. 对选中的原始 affinity 做 sigmoid，并在 top-k 内归一化。
            """
            input_dtype = hidden_states.dtype
            
            # 1. Jitter Noise (训练稳定性技巧，可选)
            if self.training and self.jitter_noise > 0:
                hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

            # 2. 原始 affinity
            router_logits = self.classifier(hidden_states)
            topk_indices, topk_weights = self._get_topk_with_bias(router_logits, k=2)
            topk_weights = topk_weights.to(input_dtype)

            # 3. 用 top-k 结果更新 batch-wise bias buffer
            if self.training:
                self._update_bias_buffer(topk_indices)

            return topk_indices, topk_weights, router_logits
    def compute_entropy_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        [Entropy Maximization Loss]
        目标：最大化路由分布的熵 (Maximize Entropy) => 最小化负熵 (Minimize Negative Entropy).
        公式：Loss = sum( p * log(p) )
        
        作用：
        1. 只要 p -> 0，log(p) 就会趋向负无穷，产生巨大梯度把 Expert 救活（防止饿死）。
        2. 在 p 正常时，它比 Load Balance 更平滑，允许 Expert 根据语义偏好（CKA）分配数据（不强求绝对平均）。
        """
        # 1. 展平维度: [Batch, Seq, Experts] -> [Batch*Seq, Experts]
        if router_logits.dim() > 2:
            router_logits = router_logits.view(-1, self.num_experts)
        
        # 2. 计算概率 (Softmax)
        probs = F.softmax(router_logits, dim=-1)
        # 3. 计算整个 Batch 的平均路由概率 (Mean Probability)
        # 这代表了每个 Expert 在当前 Batch 里“抢到了”多少比例的数据
        mean_probs = probs.mean(dim=0) 
        # 4. 计算负熵 Loss
        # 加上 1e-6 是为了防止 log(0) 导致 NaN
        # entropy_loss = (mean_probs * torch.log(mean_probs + 1e-6)).sum()
        entropy_loss = (mean_probs * torch.log(mean_probs + 1e-6)).sum()

        
        return entropy_loss
    def compute_load_balancing_loss(self,router_logits: torch.Tensor, num_experts: int, top_k: int = 2) -> torch.Tensor:
        """
        Computes the load balancing loss to encourage balanced expert utilization.
        Reference: Switch Transformers / GShard
        Loss = num_experts * sum(fraction_of_prob * fraction_of_tokens)
        """
        # 1. Flatten the batch and sequence dimensions to (batch_size * seq_len, num_experts)
        # This ensures compatibility whether input is [Batch, Seq, Experts] or [Batch*Seq, Experts]
        if router_logits.dim() > 2:
            router_logits = router_logits.view(-1, num_experts)
        
        # 2. Softmax to get probabilities (density_1_proxy)
        # P_i: The average probability assigned to expert i across the batch
        routing_probs = F.softmax(router_logits, dim=-1)
        density_1 = routing_probs.mean(dim=0)

        # 3. Get Top-K selections (density_2_proxy)
        # f_i: The fraction of tokens routed to expert i (based on top-k selection)
        _, topk_indices = torch.topk(router_logits, k=top_k, dim=-1)
        
        # Create mask: [total_tokens, top_k, num_experts] -> [total_tokens * top_k, num_experts]
        mask = F.one_hot(topk_indices, num_classes=num_experts)
        mask = mask.view(-1, num_experts).float()
        
        # Calculate the fraction of times each expert was selected
        # Note: We calculate the mean over the flattened mask. 
        # Since we select top_k experts, the sum of density_2 will be close to top_k/num_experts (scaling factor)
        # Ideally we compare expectations.
        density_2 = mask.mean(dim=0)

        # 4. Compute the dot product and scale
        # We want to minimize the dot product between P (probs) and f (frequencies)
        # Multiplied by num_experts squared? No, usually just num_experts * (sum P_i * f_i)
        loss = (density_1 * density_2).sum() * num_experts

        return loss
    def _get_topk_with_bias(self, router_logits: torch.Tensor, k: int = 2):
        """
        使用 (logits + CKA_bias + load_bias) 进行专家选择。
        对应公式 Eq.(9): s_{i,t} + alpha * H_cp^{(i)} + b_i^{(t}) ∈ TopK
        """
        # expert_bias is only applied during training (load balancing signal)
        # At inference, use pure logits + CKA to avoid forcing suboptimal uniform routing
        load_bias = self.expert_bias if self.training else torch.zeros_like(self.expert_bias)
        inference_beta = self.beta if self.training else 0.0  # In inference, we can choose to ignore the CKA bias or keep it. Here we keep it for better routing decisions.
        selection_logits = router_logits + inference_beta * self.expert_similarities + load_bias
        _, indices = torch.topk(selection_logits, k=k, dim=-1)
        topk_original_logits = router_logits.gather(dim=-1, index=indices)
        topk_scores = torch.sigmoid(topk_original_logits.to(torch.float32))
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return indices, topk_weights
    def _update_bias_buffer(self, topk_indices: torch.Tensor):
        if not self.training:
            return

        # 1. 计算当前卡的局部计数 (Local count)
        active_indices = topk_indices.view(-1)
        expert_count = torch.bincount(active_indices, minlength=self.num_experts).float()
        
        # 2. 【关键】同步全局计数 (Global sync)
        # 只有执行了这一步，各卡算出的 bias_delta 才是完全一致的
        if torch.distributed.is_initialized():
            import torch.distributed as dist
            # 汇总所有显卡的专家选择次数
            dist.all_reduce(expert_count, op=dist.ReduceOp.SUM)
            
            # 汇总总 Token 选择次数 (Batch * Seq * K)
            total_selections = torch.tensor([active_indices.numel()], 
                                        device=active_indices.device, 
                                        dtype=torch.float)
            dist.all_reduce(total_selections, op=dist.ReduceOp.SUM)
            global_numel = total_selections.item()
        else:
            global_numel = active_indices.numel()

        # 3. 计算全局占用率并更新
        current_util = expert_count / global_numel
        target_util = 1.0 / self.num_experts
        
        bias_delta = self.bias_update_rate * (target_util - current_util)
        
        # 因为上面同步了计数，所以这里的 add_ 在所有卡上增加的数值是完全一样的
        self.expert_bias.add_(bias_delta)
    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        """
        返回: (topk_indices, topk_weights, router_logits, use_te, warm_end)
        """
        self.router_aux_loss = 0.0 
        
        # =================================================
        # 场景 A: 推理模式 (Inference)
        # =================================================
        if not self.training:
            # 直接使用 Router
            topk_indices, topk_weights, router_logits = self._compute_router_probabilities(hidden_states)
            
            # use_te=False, warm_end=False
            return topk_indices, topk_weights, router_logits, False, False
        
        # =================================================
        # 场景 B: 训练模式 (Training)
        # =================================================
        
        # 1. 累积 Token 计数
        # 注意：这里计算的是当前 Batch 的总 Token 数
        batch_token_count = hidden_states.numel() // hidden_states.size(-1)
        self.processed_tokens += batch_token_count
        
        # 2. 【防死锁核心】状态同步判断
        # 先在本地判断是否达到了阈值
        local_warm_done = (self.processed_tokens >= self.threshold)
        
        # 如果已经处于 Stable 阶段，就不需要再判断了
        if bool(self.warm_end):
            is_warmup_phase = False
            trigger_warm_end = False
        else:
            # 如果还在 Warmup，检查是否该结束了
            # 为了防止不同 GPU 计数差异导致死锁，这里进行一次全归约 (AllReduce)
            # 逻辑：只要任意一张卡 (MAX) 达到了阈值，所有卡都视为达到阈值
            if torch.distributed.is_initialized():
                import torch.distributed as dist
                signal = torch.tensor([1.0 if local_warm_done else 0.0], device=hidden_states.device)
                dist.all_reduce(signal, op=dist.ReduceOp.MAX)
                global_warm_done = (signal.item() > 0.5)
            else:
                global_warm_done = local_warm_done

            if global_warm_done:
                # 刚刚跨过阈值的那一刻
                self.warm_end.fill_(True)
                is_warmup_phase = True # 这一步依然算 Warmup (或者是最后一步 TE)
                trigger_warm_end = True # 触发外部的 "离线计算" 信号
            else:
                # 还没到阈值
                is_warmup_phase = True
                trigger_warm_end = False

        # =================================================
        # 分支 1: Warmup 阶段 (使用 TE)
        # =================================================
        if is_warmup_phase:
            # 返回 use_te=True
            # 其他参数为 None，因为这时候不用 Router
            return None, None, None, True, trigger_warm_end

        # =================================================
        # 分支 2: Stable 阶段 (使用 Router + SE)
        # =================================================
        topk_indices, topk_weights, router_logits = self._compute_router_probabilities(hidden_states)
        
        # 计算负载均衡 Loss (Aux Loss)
        # 确保你的类里有 expert_num (例如 4)
        # if self.training:
        #     #  self.router_entropy_loss = self.compute_entropy_loss(router_logits)
        #      self.router_aux_loss = self.compute_load_balancing_loss(
        #          router_logits, 
        #          num_experts=self.num_experts,
        #          top_k=2
        #      )
        
        # use_te=False, warm_end=False
        return topk_indices, topk_weights, router_logits, False, False

class TransientExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int, te_dim: int):
        super().__init__()
        # Down-projection (相当于 LoRA A)
        self.down = nn.Linear(in_features, te_dim, bias=False)
        # Up-projection (相当于 LoRA B)
        self.up = nn.Linear(te_dim, out_features, bias=False)
        self.importance_mask = None
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # A (down) 使用高斯分布初始化
        nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        # B (up) 初始化为0，保证初始状态为恒等映射（无影响）
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-5)        
        # nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fix: 自动设备对齐，防止 TE 在 CPU 上导致 DeepSpeed 报错
        if self.down.weight.device != x.device:
            self.to(x.device) 
        
        # 线性路径: x -> down -> up
        return self.up(self.down(x))

class CLMoEMOEExpert(nn.Module):

    def __init__(self, in_features, out_features):
        
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight
    

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y

class CLMoEMOEGate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)    # 第0维为batch size
    
    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y

class CLMoEMOELinearA(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraA.append(CLMoEMOEExpert(self.in_features, self.r))

    
    def forward(self, x):
        '''input x is a vector, return output is a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))

        return outputs
    
class CLMoEMOELinearB(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraB.append(CLMoEMOEExpert(self.r, self.out_features))

    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))

        return outputs

class TransientSIManager:
    """
    Hook 版 SI Manager：直接在反向传播过程中拦截梯度，完美兼容 ZeRO-3。
    """
    def __init__(self, te_module: nn.Module, damping_factor: float = 0.1):
        self.te_module = te_module
        self.damping = damping_factor
        
        self.omega = {}       
        self.params_prev = {} 
        self.params_start = {}
        self.hooks = [] # 存储 hook 句柄
        
        self._init_tracking()

    def _init_tracking(self):
        for name, param in self.te_module.named_parameters():
            if param.requires_grad:
                # 1. 初始快照 (CPU)
                with zero.GatheredParameters([param], modifier_rank=None):
                    self.params_start[name] = param.data.detach().clone().cpu()
                    self.params_prev[name] = param.data.detach().clone().cpu()
                    self.omega[name] = torch.zeros_like(param.data) # 放在 GPU

                # 2. 【关键】注册 Hook
                # 只要梯度一产生，立刻触发 self._on_grad_hook
                # 使用默认参数 (name=name) 捕获闭包变量
                h = param.register_hook(lambda grad, n=name, p=param: self._on_grad_hook(grad, n, p))
                self.hooks.append(h)
    
    def _on_grad_hook(self, grad, name, param):
        """
        这个函数会在 backward 过程中被自动调用。
        grad: 当前参数的梯度 (可能是切片)
        """
        try:
            # 在 Hook 里，我们不需要 Gather 参数，因为 grad 和 param.data 肯定是设备/形状对齐的
            curr_data = param.data
            
            # 动态对齐
            if name not in self.params_prev: return
            prev_data = self.params_prev[name].to(curr_data.device)
            
            if self.omega[name].device != curr_data.device:
                self.omega[name] = self.omega[name].to(curr_data.device)

            # 1. 计算 delta
            delta = curr_data - prev_data
            # print(delta.norm().item(), flush=True)
            # 2. 累积 omega
            # 注意：grad 是正在流动的 tensor，不要改它，只读
            self.omega[name].add_(-grad * delta)
            
            # 3. 更新快照 (存回 CPU)
            self.params_prev[name] = curr_data.detach().cpu().clone()
            
        except Exception as e:
            print(f" Error in SI Hook ({name}): {e}", flush=True)


    def calculate_importance(self):
        importance_mask = {}        
        for name, param in self.te_module.named_parameters():
            if param.requires_grad and name in self.omega:                
                curr_data = param.data
                start_data = self.params_start[name].to(curr_data.device)
                omega = self.omega[name].to(curr_data.device)

                delta_total = curr_data - start_data
                denom = delta_total.pow(2) + self.damping
                importance_mask[name] = torch.nn.functional.relu(omega) / denom
                
        self.te_module.importance_mask = importance_mask
        self.cleanup()
        global_max = max(mask.max().item() for mask in importance_mask.values())
        # print(f"[Debug] Global max of importance_mask: {global_max:.6f}")

        # 2. 此处做归一化
        if global_max > 0:
            for name in importance_mask:
                importance_mask[name] = importance_mask[name] / global_max
                
        # print(f"[Debug] Global max was: {global_max:.6f}, normalized to 0~1")

        return importance_mask

    def cleanup(self):
        # 移除 hooks
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.omega = {}
        self.params_prev = {}
        self.params_start = {}
# ==========================================
# 3. Layers & Model
# ==========================================

class CLMoEMOELoraLayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, expert_num: int):
        
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        
    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: CLMoEMOELinearA(self.in_features, r, self.expert_num)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: CLMoEMOELinearB(r, self.out_features, self.expert_num)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        # Fix: 删除 self.to(self.weight.device)，避免在 ZeRO-3 中过早移动
    
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.normal_(self.lora_B[adapter_name].loraB[i].mlp.weight, mean=0.0, std=1e-5)
class CLMoEMOELoraLinear(nn.Linear, CLMoEMOELoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        
        **kwargs,
    ):
        
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", 4) # 修正: 给默认值防止报错
        self.te_dim = kwargs.pop("task_embedding_dim", 64) # 修正: 给默认值
        self.warmup_tokens = kwargs.pop('warmup_tokens', 10000) # 修正: 给默认值
        self.noisy_gating = True
        self.pending_mask_updates = {}
        self.topk = 2

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CLMoEMOELoraLayer.__init__(self, in_features=in_features, 
                               out_features=out_features, 
                               expert_num=self.expert_num)
        
        # 1. Router  TSMoERouter (Now defined BEFORE usage)
        self.lora_router = nn.ModuleDict({})
        threshold = kwargs.get("warmup_tokens", self.warmup_tokens)
        
        self.lora_router.update(nn.ModuleDict({
            adapter_name: TSMoERouter(  # 改为 TSMoERouter
                config=CLMoEMOELoraConfig( # 构造临时 config 传入，或者直接传参数
                    warmup_tokens=threshold,
                    expert_num=self.expert_num,
                    task_embedding_dim=self.te_dim  
                ),
                in_features=in_features
            )
        }))

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        # ======= 关键：用 SE 的 rank 覆盖 te_dim =======

        r_per_expert = self.lora_A[adapter_name].r  # 每个 expert 的 rank
        if self.te_dim != r_per_expert:
            self.te_dim = r_per_expert
        # 2. 初始化 TE
        self.transient_experts = nn.ModuleDict({})
        te_instance = TransientExpert(
            in_features=in_features, 
            out_features=out_features, 
            te_dim=self.te_dim
        )
        self.transient_experts.update(nn.ModuleDict({adapter_name: te_instance}))
        
        # 3. 初始化 SI Manager
        self.si_managers = {} 
        self.si_managers[adapter_name] = TransientSIManager(
            te_module=te_instance,
            damping_factor=0.1
        )
        # 4. 初始化 Expert Masks
        self.expert_masks = [{} for _ in range(self.expert_num)]
        self.anchor_params = [{} for _ in range(self.expert_num)]
        # 5. CKA 用的 hidden state 积累 buffer
        self.te_repr_buffer: List[torch.Tensor] = []
        self.te_repr_buffer_max_tokens: int = 4096  # 最多积累多少 token
    
    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
             return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
        elif self.r[self.active_adapter] > 0:

            # A. 底座输出
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].mlp.weight.dtype)
            
            # Fix: Remove self.lora_router.to(x.device) for ZeRO-3 safety
            topk_indices, topk_weights, router_logits, use_te, warm_end = self.lora_router[self.active_adapter](x)

            if use_te:
                # === Warmup Phase (TE) ===
                te_layer = self.transient_experts[self.active_adapter]
                result += te_layer(x) * self.scaling[self.active_adapter]
                # 积累 hidden states 用于后续 CKA 计算
                if self.training:
                    curr_tokens = x.view(-1, x.size(-1))
                    if sum(t.size(0) for t in self.te_repr_buffer) < self.te_repr_buffer_max_tokens:
                        self.te_repr_buffer.append(curr_tokens.detach().cpu())
                if warm_end and self.training:
                    # 触发 Warmup 结束逻辑
                    self.calculate_te_si(self.active_adapter)
                    with torch.no_grad():
                        if self.te_repr_buffer:
                            validation_data = torch.cat(self.te_repr_buffer, dim=0).to(x.device)
                            self.te_repr_buffer.clear()
                        else:
                            validation_data = x
                        self.allocate_expert_and_mask(self.active_adapter, validation_data=validation_data)        
            else:
                # === Stable Phase ===
                for i in range(self.expert_num):
                    mask = (topk_indices == i)
                    expert_routing_weight = (topk_weights * mask.float()).sum(dim=-1, keepdim=True)

                    expert_out = self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](
                            self.lora_dropout[self.active_adapter](x)
                        )
                    )
                    result += (
                        expert_out 
                        * self.scaling[self.active_adapter] 
                        * expert_routing_weight
                    )
        else:
             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result

    
    def accumulate_te_grad(self, adapter_name):
        """外部训练循环调用：积累梯度"""
        if adapter_name in self.si_managers:
            self.si_managers[adapter_name].accumulate_step()

    def calculate_te_si(self, adapter_name):
        """内部调用：计算最终 SI 矩阵"""
        if adapter_name in self.si_managers:
            self.si_managers[adapter_name].calculate_importance()

    def allocate_expert_and_mask(self, adapter_name, validation_data):
        """
        CKA 分配逻辑：增加了打印所有 Expert CKA 值的功能。
        """        
        te_module = self.transient_experts[adapter_name]
        if te_module.importance_mask is None:
            print(f"⚠️ Warning: Mask is None. Calculating it now...")
            if adapter_name in self.si_managers:
                self.calculate_te_si(adapter_name)

        omega_t = te_module.importance_mask 
        
        # --- 1. 计算 CKA ---
        te_repr = self.get_representation(te_module, validation_data)
        cka_scores = []
        for i in range(self.expert_num):
            se_repr = self.get_representation(adapter_name, i, validation_data)
            # print(se_repr.mean)
            score = linear_cka(te_repr, se_repr)
            cka_scores.append(score)
        print(f"CKA Scores for adapter '{adapter_name}': {cka_scores}")  # Debug: 打印每个 Expert 的 CKA 分数
        # 3. 【关键】多卡同步 CKA 分数 (Global Sync)
        # 将 list 转为 tensor 放入 GPU
        device = self.weight.device
        cka_tensor = torch.tensor(cka_scores, device=device, dtype=torch.float32)
        import torch.distributed as dist
        if dist.is_initialized():
            # 取所有显卡的平均值，保证大家看到的“潜能”是一样的
            dist.all_reduce(cka_tensor, op=dist.ReduceOp.AVG)
        
        final_cka_scores = cka_tensor.tolist()
        # 4. 【注入参数】更新 Router 的 Permeation Potential
        if adapter_name in self.lora_router:
            # 使用 copy_ 原地更新 buffer，不破坏计算图
            self.lora_router[adapter_name].expert_similarities.copy_(cka_tensor)
        for i in range(self.expert_num):
            similarity = final_cka_scores[i]  # 使用多卡同步后的全局平均值

            # A. 取出旧 Mask (如果不存在则初始化为空字典)
            # 注意：这里我们深拷贝一份，因为我们要修改它并存入 pending
            current_old_mask = self.expert_masks[i]
            updated_mask = copy.deepcopy(current_old_mask) if current_old_mask else {}
            
            # B. 计算叠加更新量: Delta = CKA * TE_SI
            for param_name, te_importance_tensor in omega_t.items():
                # 确保设备一致
                if param_name in updated_mask:
                    target_device = updated_mask[param_name].device
                else:
                    # 如果是新初始化的 mask，通常跟随 TE device 或默认 device
                    target_device = te_importance_tensor.device 

                te_val = te_importance_tensor.to(target_device)
                update_term = similarity * te_val
                
                # C. 执行累加 (Lithification)
                if param_name in updated_mask:
                    updated_mask[param_name] += update_term
                else:
                    updated_mask[param_name] = update_term
            
            # D. 存入 pending 队列
            # 这样 self.expert_masks 保持不变，直到训练循环显式调用 update_masks()
            self.pending_mask_updates[i] = updated_mask
        # curr_step = self.lora_router[adapter_name].processed_tokens.item()
        # safe_layer_name = getattr(self, "layer_name", f"layer_{id(self)}")
        # save_layer_masks_as_image(self.expert_masks, safe_layer_name, curr_step)
        # --- 3. 清理 ---
        if adapter_name in self.si_managers:
            del self.si_managers[adapter_name]
    def commit_masks(self):
        """
        [任务训练结束后调用]
        将 pending buffer 中的 Mask 正式应用到 expert_masks 中。
        这意味着当前任务的 Mask 将开始对下一个任务产生约束。
        """
        if not self.pending_mask_updates:
            # print("No pending masks to commit.") # 可以注释掉避免刷屏
            return
            
        
        for idx, new_mask in self.pending_mask_updates.items():
            self.expert_masks[idx] = new_mask
            
        # 清空缓冲区
        self.pending_mask_updates = {}


    
    def map_te_name_to_se(self, te_key):
    # "down.xxx" -> "lora_A.mlp.xxx"
        if te_key.startswith("down."):
            suffix = te_key[len("down."):]
            return f"lora_A.mlp.{suffix}"
        if te_key.startswith("up."):
            suffix = te_key[len("up."):]
            return f"lora_B.mlp.{suffix}"
        return te_key

    def update_all_anchors(self):
        """
        【必须在加载完上一个任务的权重后调用】
        将当前所有专家的参数状态保存为 Anchor，作为防遗忘的基准。
        """
        for i in range(self.expert_num):
            # 获取当前专家的参数引用
            lora_a = self.lora_A[self.active_adapter].loraA[i]
            lora_b = self.lora_B[self.active_adapter].loraB[i]
            
            anchor_dict = {}
            # 必须使用 .detach().clone()，否则它会随训练变化
            for name, param in lora_a.named_parameters():
                 anchor_dict[f"lora_A.{name}"] = param.detach().clone()
            for name, param in lora_b.named_parameters():
                 anchor_dict[f"lora_B.{name}"] = param.detach().clone()
                 
            # 存入存储区
            self.anchor_params[i] = anchor_dict


    def get_regularization_loss(self):
        import torch.utils.checkpoint

        # 1. 定义 EWC 计算函数 (Checkpoint 需要一个闭包或函数)
        def calc_ewc(theta, theta_ref, omega):
            # 简单的加权平方差
            return (omega * ((theta - theta_ref) ** 2)).sum()

        reg_loss = torch.tensor(0.0, device=self.weight.device)
        
        # Warmup 检查
        if self.active_adapter in self.lora_router:
            router = self.lora_router[self.active_adapter]
            if hasattr(router, 'warm_end') and not router.warm_end.item():
                return reg_loss
        for i in range(self.expert_num):
            mask_dict = self.expert_masks[i]
            anchor_dict = self.anchor_params[i]
            if not mask_dict or not anchor_dict: continue
            
            # 获取当前参数
            current_params = {}
            for name, param in self.lora_A[self.active_adapter].loraA[i].named_parameters():
                if "weight" in name: current_params["down.weight"] = param
            for name, param in self.lora_B[self.active_adapter].loraB[i].named_parameters():
                if "weight" in name: current_params["up.weight"] = param
            
            for param_name, omega in mask_dict.items():
                if param_name not in current_params: continue
                
                theta = current_params[param_name]
                
                # 匹配 Anchor 参数名
                target_key = None
                if param_name in anchor_dict: target_key = param_name
                elif "down.weight" in param_name and "lora_A.weight" in anchor_dict: target_key = "lora_A.weight"
                elif "up.weight" in param_name and "lora_B.weight" in anchor_dict: target_key = "lora_B.weight"
                if not target_key: continue

                try:
                    # 2. 准备数据 (detach 依然需要，防止计算图无限增长)
                    theta_ref = anchor_dict[target_key].to(theta.device).detach()
                    omega_val = omega.to(theta.device).detach()
                    # 3. 核心计算：直接计算，不要用 Checkpoint！
                    # DeepSpeed Zero-1 会自动处理这里的梯度累加，不会报错
                    if theta.requires_grad:
                        reg_loss += calc_ewc(theta, theta_ref, omega_val)

                    # 4. 显存清理
                    del theta_ref, omega_val

                except Exception as e:
                    print(f"Warning: EWC calc failed: {e}")
                    continue

        return reg_loss

    def get_representation(self, module_or_adapter, expert_idx_or_data, data=None):
        """
        统一获取表示的辅助函数。
        """
        with torch.no_grad():
            if isinstance(module_or_adapter, nn.Module): # TE 情况
                return module_or_adapter(expert_idx_or_data) # expert_idx_or_data 这里其实是 data
            else: # SE 情况
                adapter_name = module_or_adapter
                expert_idx = expert_idx_or_data
                # 模拟 Expert Forward: A -> Dropout -> B
                lora_a = self.lora_A[adapter_name].loraA[expert_idx]
                lora_b = self.lora_B[adapter_name].loraB[expert_idx]
                return lora_b(lora_a(data))
        

    def export_masks(self):
        """
        导出 Mask 状态，包括：
        1. expert_masks: 当前正在生效的、用于正则化的旧 Mask。
        2. pending_mask_updates: 刚刚计算出但尚未提交的新 Mask (Buffer)。
        """
        return {
            'expert_masks': self.expert_masks,
            'pending_mask_updates': self.pending_mask_updates 
        }

    def load_masks(self, state_dict):
        """
        从磁盘加载 Mask 状态。
        """
        # 1. 加载正式生效的 Mask
        if 'expert_masks' in state_dict:
            self.expert_masks = state_dict['expert_masks']
        else:
            print(f" Warning: No 'expert_masks' found for layer {getattr(self, 'layer_name', 'unknown')}")

        # 2. 加载缓冲区 (如果有的话)
        # 这对于恢复意外中断的训练非常重要
        if 'pending_mask_updates' in state_dict:
            self.pending_mask_updates = state_dict['pending_mask_updates']
        else:
            # 如果旧版 Checkpoint 没有这个字段，就初始化为空
            self.pending_mask_updates = {}

# ==========================================
# 4. Model Wrapper
# ==========================================

class CLMoEMOELoraModel(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:  # get the lora config
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_clitmoelora_config(config, model_config)   # load config
            self.peft_config[adapter_name] = config # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        # 1. 正常 PEFT 冻结
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        
        # ================== ✅ 新增：强制解冻 TE 并移动到 GPU ==================
        # 尝试获取目标设备
        try:
            target_device = next(self.model.parameters()).device
            if target_device == torch.device("cpu") and torch.cuda.is_available():
                target_device = torch.device("cuda")
        except:
            target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print(f"Re-enabling gradients for Transient Experts and moving to {target_device}...")
        
        count = 0
        for name, param in self.model.named_parameters():
            if "transient_experts" in name:
                param.requires_grad = True
                # 手动搬运
                if param.device == torch.device("cpu"):
                    param.data = param.data.to(target_device)
                    if param._grad is not None:
                        param._grad = param._grad.to(target_device)
                count += 1
        
        # 再次确保 module 级别也在 GPU
        for name, module in self.model.named_modules():
            if "transient_experts" in name or "lora_router" in name:
                module.to(target_device)

        print(f"Processed {count} TE parameters.")
        # ======================================================================
        for name, module in self.model.named_modules():
            # 判断是不是我们的自定义层
            if isinstance(module, CLMoEMOELoraLinear):
                # 直接把名字存成一个属性，方便后面画图用
                module.layer_name = name
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)


    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )

            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)
        
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
            "warmup_tokens": getattr(lora_config, "warmup_tokens", 10000),
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = CLMoEMOELoraLinear(adapter_name, in_features, out_features, 
                                            bias=bias, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


    @staticmethod
    def _prepare_clitmoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                # if merge:
                #     target.merge()
                # self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model
    