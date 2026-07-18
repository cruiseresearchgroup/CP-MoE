import json, re, os, glob
from collections import defaultdict, Counter
import numpy as np

# ---------- 工具 ----------
def find_span(tokens):
    """保留原函数（可能用于回退或调试）；不影响对外行为"""
    for i, t in enumerate(tokens):
        if "<0x0A>" in str(t):
            return i + 1, len(tokens)
    raise ValueError("没找到 <0x0A>")

def preprocess_tokens(raw_tokens):
    """
    若 JSON 中 tokens 是单个字符串或几行大块文本，
    仅按空格一次性拆成列表（保证长度为 T）。
    若已经是列表就直接返回。
    """
    if isinstance(raw_tokens, str):
        return raw_tokens.split()
    if isinstance(raw_tokens, list):
        return raw_tokens
    raise TypeError(f"不支持的 tokens 类型: {type(raw_tokens)}")

# 只关注层数的数字
def get_layer_id(name):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else None

def to_array(a):
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(f"routing 形状不对: {arr.shape}")
    return arr

# ---------- 新增：序列匹配与 mask ----------
def _find_all(tokens, pred):
    return [i for i, t in enumerate(tokens) if pred(str(t))]

def _find_seq(tokens, seq):
    """
    在 tokens 中寻找一个精确子序列 seq（元素按 str 比较），返回起始索引；找不到返回 -1。
    """
    T, L = len(tokens), len(seq)
    if L == 0 or L > T:
        return -1
    for i in range(T - L + 1):
        ok = True
        for j in range(L):
            if str(tokens[i + j]) != seq[j]:
                ok = False
                break
        if ok:
            return i
    return -1

def _build_mask(tokens):
    """
    返回长度为 T 的布尔 mask：
    1) 仅保留“第1个 <0x0A> 与第2个 <0x0A> 之间”的片段（问题部分）；
    2) 以及从 “ASSISTANT 标记” 起，到结束标记（如 '</s>' 或 ''）之前的答案部分。
    若缺失锚点，回退为旧规则：从第1个 <0x0A> 之后到序列末尾。
    """
    T = len(tokens)
    mask = [False] * T

    # 精确匹配独立的 '<0x0A>'
    nl_idx = _find_all(tokens, lambda s: s.strip() == "<0x0A>")

    def _first_end_after(pos):
        # 常见结束标记候选
        ends = []
        for tag in ("</s>", ""):
            j = _find_seq(tokens[pos:], [tag])
            if j != -1:
                ends.append(pos + j)
        return min(ends) if ends else T

    added_any = False

    # 1) 问题：第1与第2个 <0x0A> 之间
    if len(nl_idx) >= 2:
        q_l, q_r = nl_idx[0] + 1, nl_idx[1]  # [q_l, q_r)
        for t in range(max(0, q_l), min(T, q_r)):
            mask[t] = True
        added_any = added_any or (q_l < q_r)

    # # 2) 答案：从 “ASSISTANT 标记” 起到结束前
    # assistant_patterns = [
    #     ["▁A", "SS", "IST", "ANT", ":"],  # '▁A SS IST ANT :'
    # ]
    # ans_start = -1
    # for pat in assistant_patterns:
    #     s = _find_seq(tokens, pat)
    #     if s != -1:
    #         ans_start = s + len(pat)
    #         break
    # if ans_start != -1 and ans_start < T:
    #     ans_end = _first_end_after(ans_start)
    #     for t in range(ans_start, min(ans_end, T)):
    #         mask[t] = True
    #     added_any = True

    # 回退：若两段都没成功，退回旧逻辑“从首个 <0x0A> 之后到末尾”
    if not added_any and len(nl_idx) >= 1:
        start = nl_idx[0] + 1
        for t in range(start, T):
            mask[t] = True

    return mask

# ---------- 单文件：返回“原始计数”而不是最终摘要 ----------
def _count_from_json_file(path):
    """
    读取一个 json 文件，返回 per-layer 的 expert->Counter(token) 原始统计。
    使用 _build_mask 精确筛选要计数的 token 位置。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "text" not in data:
        raise KeyError(f"{path} 缺少 'text' 字段")

    tokens = preprocess_tokens(data["text"])
    T = len(tokens)

    routing = data.get("routing", {})
    layers = defaultdict(list)  # lid -> [topk_idx_per_token]
    for name, v in routing.items():
        lid = get_layer_id(name)
        if lid is None or lid > 31:
            continue
        top_k_list = v.get("topk_idx")
        if top_k_list is None:
            continue
        if len(top_k_list) != T:
            # 长度不符就跳过该路由序列
            # 这里保持与原逻辑一致：warning 后 continue
            print(f"warning: {path} | {name} 长度不符 {len(top_k_list)} != {T}")
            continue
        layers[lid].append(top_k_list)

    # 使用 mask 精确筛选计数区间
    mask = _build_mask(tokens)
    idx_iter = [i for i, m in enumerate(mask) if m]
    # 统计到原始 Counter
    per_layer_cnt = defaultdict(lambda: defaultdict(Counter))  # lid -> expert -> Counter(token)
    for lid in range(32):
        for arr in layers.get(lid, []):
            # arr[t] 是该 token 的 top-k expert 列表
            for t in idx_iter:
                tok = tokens[t]
                for e in arr[t]:
                    try:
                        ee = int(e)
                    except Exception:
                        continue
                    per_layer_cnt[lid][ee][tok] += 1

    return per_layer_cnt  # dict[int -> dict[int -> Counter]]

# ---------- 聚合多个文件，并形成与原先 analyze_json 相同的输出结构 ----------
def _summarize_counts(per_layer_cnt, topn):
    """
    per_layer_cnt: lid -> expert -> Counter(token)
    返回：lid -> expert -> {"total": int, "top": [(token, n), ...]}
    """
    res = {}
    for lid in range(32):
        experts = per_layer_cnt.get(lid, {})
        res[lid] = {
            e: {"total": sum(c.values()), "top": c.most_common(topn)}
            for e, c in experts.items()
        }
    return res

# 保留原有接口（单文件），内部用原始计数生成摘要
def analyze_json(path, topn=10):
    """从 json 文件读取并统计 32 层 expert 的前 topn 偏好"""
    per_layer_cnt = _count_from_json_file(path)
    return _summarize_counts(per_layer_cnt, topn)

# 新增：读取整个文件夹（所有 .json），聚合后输出与 analyze_json 相同结构
def analyze_folder(folder, topn=10, pattern="*.json"):
    """
    遍历 folder 下所有匹配 pattern 的 json 文件，聚合 32 度 expert 的偏好统计。
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"文件夹中未找到 {pattern}: {folder}")

    # 累加原始计数
    agg = defaultdict(lambda: defaultdict(Counter))  # lid -> expert -> Counter(token)
    for path in files:
        try:
            per_layer_cnt = _count_from_json_file(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        for lid, experts in per_layer_cnt.items():
            for e, counter in experts.items():
                agg[lid][e].update(counter)

    return _summarize_counts(agg, topn)

# ---------- 输出为 JSON（保留函数名） ----------
def show_top10(summary, max_exp=None, save_path="summary_top10.txt"):
    """
    将 summary 保存为更可读的 JSON：
    {
      "Layer 0": {
        "Expert 7": {
          "hits": 228,
          "top_tokens": [{"token": "▁the", "count": 228}, ...]
        }
      },
      ...
    }
    """
    result = {}

    for lid in range(32):
        exps = summary.get(lid, {})
        if not exps:
            continue

        layer_key = f"Layer {lid}"
        result[layer_key] = {}

        items = sorted(exps.items(), key=lambda kv: kv[0])
        if max_exp:
            items = items[:max_exp]

        for e, info in items:
            expert_key = f"Expert {e}"
            top_tokens = [{"token": tok, "count": n} for tok, n in info["top"]]
            result[layer_key][expert_key] = {
                "hits": info["total"],
                "top_tokens": top_tokens
            }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # 不打印明细，避免你看不到打印的问题；如需提示，可自行添加一行 print。

# ---------- 用法示例 ----------
if __name__ == "__main__":
    folder = "./results/CLMoE/recognition-4ep-TS/Finetune/logs"
    summary = analyze_folder(folder, topn=20)   # 读取整个文件夹
    show_top10(summary, max_exp=10, save_path="summary_top10.txt")  # 存成 JSON
