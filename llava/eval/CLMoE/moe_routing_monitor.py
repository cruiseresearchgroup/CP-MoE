# === moe_routing_monitor.py ===
import torch
import json
from collections import defaultdict


class MoERoutingMonitor:
    """
    监控 CLMoEMOELoraLinear 的 token→expert 选择。
    设计目标：
      - 在训练/全并行阶段: 还能看到一整个序列 (S>1)。
      - 在推理/生成阶段: 每次 forward 通常只有最后新增的1个token，我们也能正确对齐它的token字符串。
    """

    def __init__(self, top_k=2):
        self.top_k = top_k
        self.records = []
        self.handles = []
        self.tokens_this_step = None
        self.current_meta = None  # 样本ID

    def set_generation_state(self, tokens_this_step, meta ="None"):
        """
        在生成循环里，每步调用一次，告知当前已知的可读token序列。
        tokens_this_step: List[str]，长度 = 已生成/已喂入模型的所有token（含prompt）
        """
        self.tokens_this_step = tokens_this_step
        self.current_meta = meta

    def _hook(self, layer_name):
        def fn(module, inputs, outputs):
            try:
                router_layer = module.lora_router[module.active_adapter]

                with torch.no_grad():
                    # inputs[0] 是该层的hidden states: [B, S, H]
                    x = inputs[0]
                    B, S, H = x.shape

                    # cast到router的dtype防止精度不匹配
                    x_cast = x.to(next(router_layer.parameters()).dtype)  # [B, S, H]


                    router_outputs = router_layer(x_cast)
                    
                    if isinstance(router_outputs, tuple):
                        # 新版 TSMoERouter 返回 5 个值，logits 在第 3 个位置 (索引 2)
                        # (topk_indices, topk_weights, router_logits, use_te, warm_end)
                        router_logits = router_outputs[2]
                    else:
                        # 旧版或标准 Router 直接返回 logits
                        router_logits = router_outputs
                    # ------------------------------------------------------------------

                    router_probs  = torch.softmax(router_logits, dim=-1) # [B, S, E]
                    B2, S2, E = router_probs.shape

                    # sanity check: B,S应该和x一致
                    # 如果不一致我们直接信router_probs的维度
                    if (B2 != B) or (S2 != S):
                        print(f"[MoERoutingMonitor] shape mismatch: x {x.shape}, probs {router_probs.shape}")
                        B, S = B2, S2

                    # top-k专家
                    topk_prob, topk_idx = torch.topk(
                        router_probs, self.top_k, dim=-1
                    )  # [B, S, top_k]

                    # 熵: 衡量路由是否confident
                    token_entropy = (-router_probs * torch.log(router_probs + 1e-12)).sum(dim=-1)  # [B, S]

                    # 我们只针对 batch=1 做可读记录
                    b = 0

                    rec = {
                        "layer": layer_name,
                        "topk_idx":   topk_idx[b].detach().cpu().tolist(),   # [S, top_k]
                        # "topk_prob":  topk_prob[b].detach().cpu().tolist(),  # [S, top_k]
                        # "entropy":    token_entropy[b].detach().cpu().tolist(),  # [S]
                    }

                    # if self.current_meta is not None:
                    #     rec["meta"] = self.current_meta

                    self.records.append(rec)

            except Exception as e:
                print(f"[MoERoutingMonitor] hook failed at {layer_name}: {e}")

        return fn

    def attach(self, model):
        """
        给模型里所有 CLMoEMOELoraLinear 层挂上forward hook。
        """
        for name, module in model.named_modules():
            if module.__class__.__name__ == "CLMoEMOELoraLinear":
                h = module.register_forward_hook(self._hook(name))
                self.handles.append(h)
        print(f"[MoERoutingMonitor] Attached to {len(self.handles)} CLMoEMOELoraLinear layers.")

    def detach(self):
        """
        取消所有hook。
        """
        for h in self.handles:
            h.remove()
        self.handles = []
    def reset(self):
        self.records = []
        self.tokens_this_step = None
        self.current_meta = None
    
    # def group(self):
    #     """
    #     把累计到的records写到json，方便之后分析/画图。
    
    #     新语义：
    #     - 同一个 layer 只导出一个对象，而不是多个step的列表。
    #     - 这个对象里的 topk_idx / topk_prob / entropy 都是把该layer所有step按时间顺序拼在一起的长序列。
    #       也就是把 [len=638] 和 [len=1] 直接变成 [len=639]。
    #     - 'layer' 字段在子对象里是多余的，被去掉。
    #     """
        
    #     merged = {}  # { layer_name: { "topk_idx": [...], "topk_prob": [...], "entropy": [...] } }
    
    #     for rec in self.records:
    #         layer_name = rec["layer"]
    
    #         # 如果这个layer还没初始化，先建空容器
    #         if layer_name not in merged:
    #             merged[layer_name] = {
    #                 "topk_idx":  [],
    #                 "topk_prob": [],
    #                 "entropy":   [],
    #             }
    
    #         # 把当前这条记录的内容拼接进同一layer的容器尾部
    #         # 注意：rec["topk_idx"] 是 [S, top_k] 形式的列表(二维)
    #         #      extend 会按行追加，从而做到累积token
    #         merged[layer_name]["topk_idx"].extend(rec["topk_idx"])
    #         merged[layer_name]["topk_prob"].extend(rec["topk_prob"])
    #         merged[layer_name]["entropy"].extend(rec["entropy"])
    
    #     return merged
    def group(self):
        """
        把累计到的records写到json，方便之后分析/画图。
    
        新语义：
        - 同一个 layer 只导出一个对象，而不是多个step的列表。
        - 这个对象里的 topk_idx / topk_prob / entropy 都是把该layer所有step按时间顺序拼在一起的长序列。
          也就是把 [len=638] 和 [len=1] 直接变成 [len=639]。
        - 'layer' 字段在子对象里是多余的，被去掉。
        """
        
        merged = {}  # { layer_name: { "topk_idx": [...], "topk_prob": [...], "entropy": [...] } }
    
        for rec in self.records:
            layer_name = rec["layer"]
    
            # 如果这个layer还没初始化，先建空容器
            if layer_name not in merged:
                merged[layer_name] = {
                    "topk_idx":  [],
                }
    
            # 把当前这条记录的内容拼接进同一layer的容器尾部
            # 注意：rec["topk_idx"] 是 [S, top_k] 形式的列表(二维)
            #      extend 会按行追加，从而做到累积token
            merged[layer_name]["topk_idx"].extend(rec["topk_idx"])
    
        return merged


