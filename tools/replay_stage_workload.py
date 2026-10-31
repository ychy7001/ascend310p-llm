#!/usr/bin/env python3
"""Replay a staged LLM workload by token shape and report wall-clock time.

The workload JSON records, per recording, the calls of one session as
{"stage", "prompt_tokens", "output_tokens"}. Within a session the stages run in
order and the calls of a stage run concurrently; several sessions can run at
once, sharing one global concurrency limit, as an application gateway would.
Each call goes to the backend with the fewest in-flight requests.

Prompts are synthetic Chinese text tokenized by the server once and saved, so
every configuration replays identical token IDs. Output length is forced with
ignore_eos and checked against usage. Each run uses its own cache_salt, so no
prefix-cache reuse is carried over between runs.

Example:
  python3 tools/replay_stage_workload.py --workload workload.json \
    --recording meeting --sessions 2 --urls http://127.0.0.1:18031 --out two.json
"""
import argparse
import concurrent.futures
import json
import pathlib
import random
import threading
import time
import urllib.request


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def synthetic_text(seed):
    rng = random.Random(seed)
    lines = [
        f"第{j}项讨论：负责人{rng.choice(['李明', '王芳', '张强', '陈洁'])}汇报"
        f"{rng.choice(['预算', '研发', '采购', '测试', '交付'])}进展为{rng.randrange(1, 100)}%，"
        f"计划在11月{rng.randrange(1, 29)}日完成，预算{rng.randrange(1, 900)}万元。"
        for j in range(500)
    ]
    return "请根据以下会议记录整理决策和行动项，保留人员、数字与日期。\n" + "\n".join(lines)


def load_prompts(path, url, model, recording, sessions, calls):
    prompts = json.loads(path.read_text()) if path.exists() else {}
    for s in range(sessions):
        for i, call in enumerate(calls):
            key = f"{recording}:{s}:{i}"
            if key in prompts:
                continue
            body = {"model": model, "prompt": synthetic_text(key), "add_special_tokens": False}
            with post(url + "/tokenize", body) as resp:
                tokens = json.load(resp)["tokens"]
            if len(tokens) < call["prompt_tokens"]:
                raise SystemExit(f"{key}: synthetic prompt has only {len(tokens)} tokens")
            prompts[key] = tokens[: call["prompt_tokens"]]
    path.write_text(json.dumps(prompts))
    return prompts


def stream_completion(url, payload):
    first = last = usage = None
    with post(url + "/v1/completions", payload) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            if any(c.get("text") for c in chunk.get("choices", [])):
                last = time.perf_counter()
                first = first or last
    return first, last, usage


def active_union(intervals):
    total = end = 0.0
    for left, right in sorted(intervals):
        total += max(0.0, right - max(left, end))
        end = max(end, right)
    return total


def spec_counters(url):
    """Sum vLLM speculative-decoding counters; empty when MTP is off."""
    totals = {}
    with urllib.request.urlopen(url + "/metrics", timeout=10) as resp:
        for line in resp.read().decode().splitlines():
            name = line.split("{")[0].split(" ")[0]
            if name in ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"):
                totals[name] = totals.get(name, 0.0) + float(line.split()[-1])
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workload", type=pathlib.Path, required=True)
    p.add_argument("--recording", required=True)
    p.add_argument("--sessions", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=3, help="global in-flight limit across all backends")
    p.add_argument("--urls", nargs="+", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", type=pathlib.Path, default=pathlib.Path("replay-prompts.json"))
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()

    calls = json.loads(a.workload.read_text())["recordings"][a.recording]["calls"]
    stages = list(dict.fromkeys(c["stage"] for c in calls))
    urls = [u.rstrip("/") for u in a.urls]
    prompts = load_prompts(a.prompts, urls[0], a.model, a.recording, a.sessions, calls)

    gate = threading.Semaphore(a.concurrency)
    lock = threading.Lock()
    in_flight = [0] * len(urls)
    rows = []
    before = [spec_counters(u) for u in urls]
    start = time.perf_counter()

    def invoke(s, i, call):
        with gate:
            with lock:
                backend = min(range(len(urls)), key=in_flight.__getitem__)
                in_flight[backend] += 1
            begin = time.perf_counter()
            try:
                payload = {
                    "model": a.model, "prompt": prompts[f"{a.recording}:{s}:{i}"],
                    "max_tokens": call["output_tokens"], "ignore_eos": True, "temperature": 0, "seed": 0,
                    "stream": True, "stream_options": {"include_usage": True},
                    "cache_salt": f"{a.out.stem}:{s}:{i}",
                }
                first, last, usage = stream_completion(urls[backend], payload)
            finally:
                with lock:
                    in_flight[backend] -= 1
        got = (usage or {}).get("prompt_tokens"), (usage or {}).get("completion_tokens")
        if got != (call["prompt_tokens"], call["output_tokens"]):
            raise RuntimeError(f"session {s} call {i}: usage {got} does not match workload")
        row = {"session": s, "call": i, "stage": call["stage"], "backend": backend,
               "start_s": begin - start, "first_s": first - start, "last_s": last - start,
               "end_s": time.perf_counter() - start, "output_tokens": got[1]}
        with lock:
            rows.append(row)

    def run_session(s):
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            for stage in stages:
                futures = [pool.submit(invoke, s, i, c) for i, c in enumerate(calls) if c["stage"] == stage]
                for f in futures:
                    f.result()
        return time.perf_counter() - start

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.sessions) as pool:
        session_s = list(pool.map(run_session, range(a.sessions)))
    elapsed = time.perf_counter() - start

    drafted = accepted = 0.0
    for url, b in zip(urls, before):
        after = spec_counters(url)
        drafted += after.get("vllm:spec_decode_num_draft_tokens_total", 0) - b.get("vllm:spec_decode_num_draft_tokens_total", 0)
        accepted += after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - b.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    output = sum(r["output_tokens"] for r in rows)
    decode_s = active_union((r["first_s"], r["last_s"]) for r in rows)
    summary = {
        "recording": a.recording, "sessions": a.sessions, "concurrency": a.concurrency, "urls": urls,
        "elapsed_s": round(elapsed, 2), "session_s": [round(x, 2) for x in session_s],
        "output_tokens": output,
        "output_tps": round(output / elapsed, 2),
        # Decode rate over the union of first-to-last-token spans; still includes prefill stalls of other calls.
        "active_decode_tps": round(sum(r["output_tokens"] - 1 for r in rows) / decode_s, 2),
        "mtp_acceptance": round(accepted / drafted, 4) if drafted else None,
        "calls": sorted(rows, key=lambda r: (r["session"], r["call"])),
    }
    a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "calls"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
