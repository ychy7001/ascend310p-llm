# SPDX-License-Identifier: Apache-2.0
"""统计真实 Prefill 每个 chunk 的专家占用；不推算端到端加速。"""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


def summarize(rows):
    hist = Counter()
    active = []
    maxima = []
    tokens = Counter()
    modules = set()
    for row in rows:
        counts = row["counts"]
        assert len(counts) == row["num_experts"]
        assert sum(counts) == row["chunk_tokens"] * row["top_k"]
        assert not row["timing_valid"]
        hist.update(counts)
        active.append(sum(value > 0 for value in counts))
        maxima.append(max(counts))
        tokens[row["chunk_tokens"]] += 1
        modules.add((row["pid"], row["module_ordinal"]))
    total_routes = sum(size * frequency for size, frequency in hist.items())
    return {
        "scope": "真实路由计数；诊断含CPU同步，不是性能测量；模块编号为诊断首次出现顺序",
        "chunks": len(rows), "module_count": len(modules),
        "chunk_token_histogram": dict(sorted(tokens.items())),
        "expert_rows_histogram": dict(sorted(hist.items())),
        "active_experts_median": median(active),
        "max_expert_rows_median": median(maxima),
        "max_expert_rows": max(maxima),
        "total_routes_across_layers": total_routes,
        "routes_in_experts_at_least": {
            str(threshold): {
                "routes": sum(size * frequency for size, frequency in hist.items() if size >= threshold),
                "fraction": sum(size * frequency for size, frequency in hist.items() if size >= threshold) / total_routes,
            } for threshold in (32, 64, 128, 256, 512)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    if not rows:
        raise ValueError("空路由采样")
    result = summarize(rows)
    with args.out.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
