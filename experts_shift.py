import json
from collections import defaultdict

KINDS = ("up_proj", "down_proj", "gate_proj")
# LIMIT = 32 * 3 * 200  # 960

def build_map(records):
    mp = defaultdict(list)
    for x in records:
        layer = x.get("layer")
        tk = x.get("topk_idx")
        if isinstance(layer, str) and isinstance(tk, list):
            for p in tk:
                if isinstance(p, list) and len(p) == 2:
                    mp[layer].append(p)
    return mp

# task list for VQA 
TASKS = ["location", "judge", "commonsense", "count"]

BASELINE_PATH = "./results/CLMoE/recognition/Finetune/routing_4ep.json"

for task in TASKS:
    NEW_PATH = f"./results/CLMoE/{task}/Finetune/routing_4ep.json"

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(NEW_PATH, "r", encoding="utf-8") as f:
        new = json.load(f)

    base_map = build_map(base)
    new_map  = build_map(new)

    shift = {k: 0 for k in KINDS}
    total = {k: 0 for k in KINDS}
    for layer, tb in base_map.items():
        tn = new_map.get(layer)
        if tn is None:
            continue
        kind = next((k for k in KINDS if layer.endswith(k)), None)
        if kind is None:
            continue
        len_b, len_n = len(tb), len(tn)
        m = min(len_b, len_n)
        for i in range(m):
            b0, b1 = tb[i]
            n0, n1 = tn[i]
    
            if set((b0, b1)) == set((n0, n1)):
                continue

            drift = 0
            if b0 != n0:
                drift += 1
            if b1 != n1:
                drift += 1
            shift[kind] += drift
        total[kind] += m * 2
    print(f"\n=== {task} shift rate ===")
    overall_shift = 0
    overall_total = 0
    for k in KINDS:
        s, t = shift[k], total[k]
        overall_shift += s
        overall_total += t
        pct = (s / t * 100) if t else 0.0
        print(f"{k:9s}: {pct:8.4f}%  ({s:,}shifts)")

    overall_pct = (overall_shift / overall_total * 100) if overall_total else 0.0
    # print(f"TOTAL    : {overall_pct:8.4f}%  ({overall_shift:,}/{overall_total:,})")
    print(f"TOTAL    : {overall_pct:8.4f}%  ({overall_shift:,}shifts)")

