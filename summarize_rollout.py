"""从 verl 的 rollout dump (jsonl) 里统计每步 reward，用来确认训练链路真的在打分。

关了 wandb 只留 console 时，step 级 metrics 不进日志，但 rollout_data_dir
里每步都有一个 <step>.jsonl，含 score 字段。
"""

import glob
import json
import os
import statistics
import sys


def main(run_dir: str) -> None:
    files = sorted(
        glob.glob(os.path.join(run_dir, "*.jsonl")),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    )
    if not files:
        sys.exit(f"no jsonl under {run_dir}")

    print(f"{'step':>4s} {'rows':>5s} {'reward_mean':>11s} {'std':>7s} {'nonzero':>7s} {'resp_len':>8s}")
    for f in files:
        step = os.path.basename(f).split(".")[0]
        scores, lens, sources = [], [], []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                for key in ("score", "reward", "acc"):
                    if key in d and isinstance(d[key], (int, float)):
                        scores.append(float(d[key]))
                        break
                out = d.get("output") or d.get("response") or ""
                if isinstance(out, str):
                    lens.append(len(out))
                if "data_source" in d:
                    sources.append(d["data_source"])
        if not scores:
            # 打印一行样例的键名，便于定位字段
            with open(f) as fh:
                first = json.loads(fh.readline())
            print(f"{step:>4s} keys={sorted(first)}")
            continue
        nz = sum(1 for s in scores if s != 0) / len(scores)
        print(
            f"{step:>4s} {len(scores):5d} {statistics.fmean(scores):11.4f} "
            f"{(statistics.pstdev(scores) if len(scores) > 1 else 0):7.4f} "
            f"{nz:7.1%} {(statistics.fmean(lens) if lens else 0):8.0f}"
        )
        if sources:
            from collections import Counter

            print("      by domain:", dict(Counter(sources)))


if __name__ == "__main__":
    main(sys.argv[1])
