# SPDX-License-Identifier: Apache-2.0
"""在诊断服务就绪后采集三类8K输入的真实路由；请求耗时不可用于性能比较。"""
import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path


def post(base, path, body):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18024")
    parser.add_argument("--container", default="qwen36-tp1-48g-112k")
    parser.add_argument("--trace", default="/tmp/route-occupancy.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "technical": "Explain how quantization affects memory bandwidth and numerical precision. ",
        "meeting": "产品会议记录：李明提出单芯片48G方案，王芳负责测试112K上下文。团队决定先保留实验基线，验证精度后再发布。陈强负责整理性能数据。风险是长上下文速度下降和量化误差。下周一复核。\n",
        "code": "def merge_intervals(intervals):\n    intervals = sorted(intervals)\n    result = []\n    for start, end in intervals:\n        if result and start <= result[-1][1]:\n            result[-1][1] = max(result[-1][1], end)\n        else:\n            result.append([start, end])\n    return result\n",
    }
    marker = args.trace + ".enabled"
    metadata = []
    args.out.mkdir(parents=True, exist_ok=True)
    for case, text in cases.items():
        tokens = post(args.base_url, "/tokenize", {
            "model": "qwen36-w4a8-c8", "prompt": text * 2048,
            "add_special_tokens": False,
        })["tokens"][:8192]
        assert len(tokens) == 8192
        subprocess.run(["podman", "exec", args.container, "touch", marker], check=True)
        try:
            response = post(args.base_url, "/v1/completions", {
                "model": "qwen36-w4a8-c8", "prompt": tokens,
                "temperature": 0, "max_tokens": 8, "ignore_eos": True,
            })
        finally:
            subprocess.run(["podman", "exec", args.container, "rm", "-f", marker], check=True)
        target = args.out / f"{case}.jsonl"
        if target.exists():
            raise FileExistsError(target)
        subprocess.run(["podman", "cp", f"{args.container}:{args.trace}", str(target)], check=True)
        subprocess.run(["podman", "exec", args.container, "rm", args.trace], check=True)
        metadata.append({"case": case, "prompt_tokens": len(tokens),
                         "prompt_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                         "usage": response["usage"], "output": response["choices"],
                         "timing_valid": False})
        print(json.dumps(metadata[-1], ensure_ascii=False), flush=True)
    with (args.out / "requests.json").open("x") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
