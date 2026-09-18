#!/usr/bin/env python3
"""汇总 run_eval_experts.sh 的结果，出横向对比表。

结果散在每个模型自己的目录下（eval_n<N>/0/<benchmark>.jsonl），
本脚本直接从 jsonl 重算 acc，不依赖日志里的 val-core 行。

用法:
    python summarize_eval.py            # VAL_N=16
    python summarize_eval.py --n 4
    python summarize_eval.py --md       # markdown 表格
"""
import argparse
import json
import os
import statistics

B = "/taijifs_sz34/share_304169095/hunyuan/yongcanyu"
EXPERTS = f"{B}/code/LLM-Fusion/teacher_experts_fp32/hf_experts_fp32_4b"
BASE = f"{B}/checkpoints/Qwen/Qwen3-4B-Instruct-2507"

# (标签, 模型目录)  —— 顺序即表格列序
MODELS = [
    ("base", BASE),
    ("math", f"{EXPERTS}/math"),
    ("science", f"{EXPERTS}/science"),
    ("code", f"{EXPERTS}/code"),
    ("if", f"{EXPERTS}/if"),
    ("agent", f"{EXPERTS}/agent"),
]

# benchmark -> 它是哪个域的主场（用于标注对角线）
BENCH = [
    ("aime2025", "math"),
    ("aime2026", "math"),
    ("gpqa", "science"),
    ("livecodebench_v5", "code"),
    ("livecodebench_v6", "code"),
    ("ifeval", "if"),
    ("ifbench", "if"),
    ("bfcl_v3_multi_turn", "agent"),
]

MAX_RESP = 16384


def load(path):
    """返回 (acc, n, 截断率)；文件缺失返回 None。"""
    if not os.path.exists(path):
        return None
    accs, lens = [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            a = r.get("acc")
            if a is None:
                a = r.get("score")
            if a is not None:
                accs.append(float(a))
            if "response_length" in r:
                lens.append(r["response_length"])
    if not accs:
        return None
    trunc = sum(1 for x in lens if x >= MAX_RESP) / len(lens) if lens else 0.0
    return statistics.mean(accs), len(accs), trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="VAL_N，决定读哪个 eval_n<N> 目录")
    ap.add_argument("--md", action="store_true", help="输出 markdown")
    args = ap.parse_args()

    # data[bench][model] = (acc, n, trunc)
    data = {b: {} for b, _ in BENCH}
    present = []
    for label, root in MODELS:
        d = f"{root}/eval_n{args.n}/0"
        got = False
        for b, _ in BENCH:
            r = load(f"{d}/{b}.jsonl")
            if r:
                data[b][label] = r
                got = True
        if got:
            present.append(label)

    if not present:
        print(f"还没有 eval_n{args.n} 结果。评测可能仍在跑：")
        print("  tail -f " + f"{B}/logs/llm_fusion/eval_experts/driver.log")
        return

    sep = " | " if args.md else "  "
    head = f"{'benchmark':<22}" + sep.join(f"{m:>9}" for m in present)
    if args.md:
        print("| benchmark | " + " | ".join(present) + " |")
        print("|---" * (len(present) + 1) + "|")
    else:
        print(f"\nVAL_N={args.n}   (★ = 该 benchmark 对应域的主场)\n")
        print(head)
        print("-" * len(head))

    for b, home in BENCH:
        cells = []
        for m in present:
            r = data[b].get(m)
            if not r:
                cells.append(f"{'--':>9}")
                continue
            acc, _, trunc = r
            mark = "★" if m == home else " "
            warn = "!" if trunc > 0.2 else ""
            cells.append(f"{acc * 100:>7.2f}{mark}{warn}")
        if args.md:
            print(f"| {b} | " + " | ".join(c.strip() for c in cells) + " |")
        else:
            print(f"{b:<22}" + sep.join(cells))

    # 均值
    means = []
    for m in present:
        v = [data[b][m][0] for b, _ in BENCH if m in data[b]]
        means.append(f"{statistics.mean(v) * 100:>7.2f} " if v else f"{'--':>9}")
    if args.md:
        print("| **mean** | " + " | ".join(x.strip() for x in means) + " |")
    else:
        print("-" * len(head))
        print(f"{'mean':<22}" + sep.join(means))

    # 覆盖度与截断告警
    print("\n覆盖度:")
    for m in present:
        done = sum(1 for b, _ in BENCH if m in data[b])
        ns = {data[b][m][1] for b, _ in BENCH if m in data[b]}
        print(f"  {m:<9} {done}/8 benchmark, 样本数 {sorted(ns)}")

    hi = [
        (m, b, data[b][m][2])
        for b, _ in BENCH
        for m in present
        if m in data[b] and data[b][m][2] > 0.2
    ]
    if hi:
        print("\n⚠️  截断率 >20%（分数被压低，不代表能力）:")
        for m, b, t in sorted(hi, key=lambda x: -x[2]):
            print(f"  {m:<9} {b:<22} {t:.0%}")


if __name__ == "__main__":
    main()
