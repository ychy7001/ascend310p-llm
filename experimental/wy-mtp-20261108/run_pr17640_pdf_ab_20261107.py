"""隔离的 310P Torch/native WY PDF 冷 Prefill A/B 窗口。"""
import argparse
import base64
import copy
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_IMAGE = "registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38"
CANDIDATE_IMAGE = "localhost/qwen36-pr17640-wy-310p:20261107"
MODEL = "Qwen3.6-35B-A3B-w8a8"
BASE = "http://127.0.0.1:18126"
PORT = "18126"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def snapshot():
    commands = {
        "rootless": ["podman", "ps", "-q"],
        "rootful": ["sudo", "-n", "podman", "ps", "-q"],
        "primary": ["systemctl", "is-active", "llm-backend-primary.service"],
        "image": ["systemctl", "--user", "is-active", "qwen36-image.service"],
        "npu": ["npu-smi", "info"],
        "memory": ["free", "-g"],
    }
    return {
        key: {"code": result.returncode, "stdout": result.stdout.strip()}
        for key, command in commands.items()
        for result in [subprocess.run(command, capture_output=True, text=True, timeout=30)]
    }


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as response:
        raw = response.read().decode()
    names = {
        "vllm:request_prefill_time_seconds_count": 0.0,
        "vllm:request_prefill_time_seconds_sum": 0.0,
        "vllm:request_decode_time_seconds_sum": 0.0,
        "vllm:prefix_cache_queries_total": 0.0,
        "vllm:prefix_cache_hits_total": 0.0,
    }
    found = set()
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if name in names:
            names[name] += float(line.split()[-1])
            found.add(name)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"missing metrics: {sorted(missing)}")
    return names


def wait_health(container):
    for _ in range(240):
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            pass
        state = subprocess.check_output(
            ["podman", "inspect", "--format", "{{.State.Status}}", container], text=True
        ).strip()
        if state == "exited":
            raise RuntimeError("model container exited during startup")
        time.sleep(5)
    raise TimeoutError("model health timeout")


def launch(root, out, arm):
    directory = out / arm
    directory.mkdir()
    manifest = copy.deepcopy(json.loads((root / "production-manifest.json").read_text()))
    command = manifest["command"]
    command[command.index("--host") + 1] = "127.0.0.1"
    command[command.index("--port") + 1] = PORT
    native = arm == "B"
    if native:
        command.extend(["--compilation-config", "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2]}"])
    manifest["image"] = CANDIDATE_IMAGE if native else BASE_IMAGE
    write(directory / "manifest.json", manifest)
    (directory / "cache").mkdir()
    tools = root / "wy-boundary-20261106" / "tools"
    mounts = [
        (root / "production-w8a8/model", "/models/Qwen3.6-35B-A3B-w8a8", "ro"),
        (directory, "/experiment", "rw"),
        (directory / "cache", "/root/.cache", "rw"),
        (tools / "start-production-replica.sh", "/workspace/start_vllm_ascend.sh", "ro"),
        (directory / "manifest.json", "/workspace/backend-primary.manifest.json", "ro"),
    ]
    for path in (
        "/usr/local/bin/npu-smi",
        "/etc/ascend_install.info",
        "/usr/local/Ascend/driver/version.info",
        "/usr/local/Ascend/driver/lib64",
        "/usr/local/dcmi",
    ):
        mounts.append((Path(path), path, "ro"))
    container = f"wy-pdf-{arm.lower()}-{time.time_ns()}"
    commandline = [
        "podman", "run", "-d", "--pull=never", "--name", container,
        "--network", "host", "--group-add", "keep-groups", "--ipc", "shareable",
        "--shm-size", "32g", "--env", "QWEN36_C8_ENABLED=0",
    ]
    if native:
        commandline += ["--env", "PYTHONPATH=/controltools:/native-wy/tools"]
    for device in ("davinci0", "davinci1", "davinci_manager", "devmm_svm", "hisi_hdc"):
        commandline += ["--device", "/dev/" + device]
    for source, target, mode in mounts:
        commandline += ["-v", f"{source}:{target}:{mode}"]
    commandline += ["--entrypoint", "/bin/bash", manifest["image"], "/workspace/start_vllm_ascend.sh"]
    write(directory / "launch-command.json", commandline)
    result = subprocess.run(commandline, capture_output=True, text=True, timeout=60, check=True)
    (directory / "launch.stdout").write_text(result.stdout + result.stderr)
    wait_health(container)
    return container


def stop(container, directory):
    subprocess.run(["podman", "stop", "--time", "30", container], capture_output=True, check=True, timeout=60)
    with (directory / "server.log").open("x") as log:
        subprocess.run(["podman", "logs", container], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    detail = json.loads(subprocess.check_output(["podman", "inspect", container], text=True, timeout=30))
    write(directory / "container-inspect.json", detail)
    assert detail[0]["State"]["Status"] == "exited"
    assert not detail[0]["State"]["OOMKilled"]


def request(case, inputs, prompt, arm, repeat):
    image = inputs / case["image"]
    encoded = base64.b64encode(image.read_bytes()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"pdf-wy-ab-20261107-{arm}-{case['id']}-{repeat}",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload, ensure_ascii=False).encode()
    start_metrics = metrics()
    start = time.monotonic()
    first = None
    first_content = None
    text = []
    reasoning = []
    usage = None
    finish = None
    request_obj = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    stream_path = arm + "-" + case["id"] + f"-{repeat}-stream.jsonl"
    with urllib.request.urlopen(request_obj, timeout=900) as response:
        with (Path(os.environ["PDF_AB_OUT"]) / stream_path).open("x") as raw:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                value = line[6:].strip()
                if value == b"[DONE]":
                    break
                event = json.loads(value)
                elapsed = time.monotonic() - start
                raw.write(json.dumps({"elapsed_s": elapsed, "event": event}, ensure_ascii=False) + "\n")
                raw.flush()
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    think = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if first is None and (content or think):
                        first = elapsed
                    if first_content is None and content:
                        first_content = elapsed
                    text.append(content)
                    reasoning.append(think)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    end = time.monotonic()
    end_metrics = metrics()
    delta = {key: end_metrics[key] - start_metrics[key] for key in start_metrics}
    result = {
        "id": case["id"], "arm": arm, "repeat": repeat,
        "image_sha": case["image_sha256"], "payload_sha": hashlib.sha256(body).hexdigest(),
        "first_delta_s": first, "first_content_s": first_content, "elapsed_s": end - start,
        "usage": usage, "finish_reason": finish, "text": "".join(text),
        "reasoning": "".join(reasoning), "metrics_delta": delta,
        "prefill_seconds": delta["vllm:request_prefill_time_seconds_sum"],
        "prefix_cache_queries": delta["vllm:prefix_cache_queries_total"],
        "prefix_cache_hits": delta["vllm:prefix_cache_hits_total"],
    }
    result["valid"] = bool(
        usage and usage.get("prompt_tokens", 0) > 0 and result["prefix_cache_hits"] == 0
        and result["finish_reason"] in ("stop", "length") and result["text"]
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    assert len(manifest["cases"]) == 5 and manifest["manual_ground_truth"] is False
    for case in manifest["cases"]:
        assert sha(args.inputs / case["image"]) == case["image_sha256"]
    atomic(args.out / "status.json", {"state": "waiting_npu_lock", "unix_ns": time.time_ns()})
    freeze = {
        "controller_sha256": sha(Path(__file__)),
        "input_manifest_sha256": sha(args.inputs / "manifest.json"),
        "source_manifest_sha256": sha(args.root / "production-manifest.json"),
        "native_candidate": "wy4-candidate-v3",
        "native_artifacts_sha256": sha(args.root / "wy4-candidate-v3/operator-artifacts.json"),
        "images": {"A": BASE_IMAGE, "B": CANDIDATE_IMAGE},
        "model": MODEL,
        "arms": ["A", "B"],
        "cases": [case["id"] for case in manifest["cases"]],
        "repeats": 1,
        "max_tokens": 512,
        "cold_prefix": True,
        "scope": "PDF visual cold-Prefill A/B diagnostic only; A=Torch WY, B=current native WY; no quality gate, no adoption evidence",
    }
    write(args.out / "freeze.json", freeze)
    started = time.monotonic()
    start_ns = time.time_ns()
    error = None
    rows = []
    container = None
    os.environ["PDF_AB_OUT"] = str(args.out)
    try:
        with open("/tmp/a310-npu.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before = snapshot()
            write(args.out / "window-start.json", before)
            assert before["rootless"]["stdout"] == before["rootful"]["stdout"] == ""
            assert before["primary"]["stdout"] == before["image"]["stdout"] == "inactive"
            for arm in ("A", "B"):
                atomic(args.out / "status.json", {"state": "launching", "arm": arm, "unix_ns": time.time_ns()})
                container = launch(args.root, args.out, arm)
                try:
                    for case in manifest["cases"]:
                        atomic(args.out / "status.json", {"state": "request", "arm": arm, "case": case["id"], "unix_ns": time.time_ns()})
                        result = request(case, args.inputs, manifest["prompt"], arm, 0)
                        write(args.out / f"{arm}-{case['id']}-result.json", result)
                        rows.append({key: value for key, value in result.items() if key not in ("text", "reasoning")})
                finally:
                    stop(container, args.out / arm)
                    container = None
            paired = []
            for case in manifest["cases"]:
                a = next(row for row in rows if row["arm"] == "A" and row["id"] == case["id"])
                b = next(row for row in rows if row["arm"] == "B" and row["id"] == case["id"])
                paired.append({
                    "id": case["id"],
                    "A_ttft_s": a["first_content_s"], "B_ttft_s": b["first_content_s"],
                    "A_prefill_seconds": a["prefill_seconds"], "B_prefill_seconds": b["prefill_seconds"],
                    "ttft_improvement": 1 - b["first_content_s"] / a["first_content_s"] if a["first_content_s"] and b["first_content_s"] else None,
                    "prefill_improvement": 1 - b["prefill_seconds"] / a["prefill_seconds"] if a["prefill_seconds"] and b["prefill_seconds"] else None,
                    "A_valid": a["valid"], "B_valid": b["valid"],
                })
            write(args.out / "summary.json", {"scope": freeze["scope"], "rows": rows, "paired": paired})
    except BaseException as exc:
        error = repr(exc)
    finally:
        if container:
            try:
                stop(container, args.out / "emergency")
            except Exception:
                pass
        after = snapshot()
        clean = (
            after["rootless"]["stdout"] == after["rootful"]["stdout"] == ""
            and after["primary"]["stdout"] == after["image"]["stdout"] == "inactive"
            and "No running processes" in after["npu"]["stdout"]
        )
        write(args.out / "window-closed.json", {
            "start_unix_ns": start_ns, "end_unix_ns": time.time_ns(),
            "elapsed_s": time.monotonic() - started, "cleanup_passed": clean,
            "snapshot": after, "error": error,
        })
        atomic(args.out / "status.json", {"state": "failed" if error else "complete", "error": error, "unix_ns": time.time_ns()})
        assert clean, "cleanup failed"
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
