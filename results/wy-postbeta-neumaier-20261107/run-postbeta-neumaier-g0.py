"""Run the bounded 612-case device G0 for the postbeta plus Neumaier candidate."""
import fcntl
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path

r = Path(__file__).resolve().parent
base = r.parent
out = r / "g0-fixed-inputs"
tools = r / "tools"
wy = base / "wy-postbeta-neumaier-v1"
candidate_sha = "e53600af2172159338673fc15c153fd5b40224265048ab41108af9de4804503e"


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


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
        key: {"code": proc.returncode, "stdout": proc.stdout.strip()}
        for key, command in commands.items()
        for proc in [subprocess.run(command, capture_output=True, text=True, timeout=30)]
    }


def state(phase, **details):
    temporary = out / "status.tmp"
    temporary.write_text(json.dumps({"phase": phase, "unix_ns": time.time_ns(), **details}, indent=2))
    temporary.replace(out / "status.json")


def interrupted(*_):
    raise TimeoutError("bounded device G0 deadline")


assert json.loads((r / "status.json").read_text())["state"] == "built_requires_g0"
assert hashlib.sha256((wy / "source/csrc/moe/chunk_gated_delta_rule_compute_wy/op_kernel/arch20/compute_wy_kernel.h").read_bytes()).hexdigest() == candidate_sha
assert json.loads((r / "build-window-closed.json").read_text())["cleanup_passed"]
out.mkdir(exist_ok=False)
container = None
error = None
started = None

with open("/tmp/a310-npu.lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    started = time.time_ns()
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.alarm(900)
    try:
        before = snapshot()
        write(out / "window-start.json", before)
        assert before["rootless"]["stdout"] == before["rootful"]["stdout"] == ""
        assert before["primary"]["stdout"] == before["image"]["stdout"] == "inactive"

        captures = base / "wy921-localize-v1/capture/captures"
        triggers = base / "wy4-candidate-v3/triggers"
        paths = [*captures.glob("rank*-call*.pt"), *triggers.glob("rank*-call*.pt")]
        assert len(paths) == 132
        write(
            out / "freeze.json",
            {
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "candidate_kernel_sha256": candidate_sha,
                "wy_manifest_sha256": hashlib.sha256((wy / "operator-artifacts.json").read_bytes()).hexdigest(),
                "input_sha256": {str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                "scope": "612 WY G0 cases: 132 snapshots/triggers, 480 boundary samples, 30 synthetic H/O cases; no model/performance acceptance",
                "action_budget_s": 900,
                "window_limit_s": 10800,
            },
        )

        template = json.loads((base / "wy4-g0-v3/launch-command.json").read_text())
        container = "wy-postbeta-neumaier-g0-" + str(time.time_ns())
        command = template
        command[command.index("--name") + 1] = container
        command[command.index(str(base / "wy4-tools") + ":/tools:ro")] = str(tools) + ":/tools:ro"
        command[command.index(str(base / "wy4-candidate-v3") + ":/wy:ro")] = str(wy) + ":/wy:ro"
        output = "/runtime/" + str(out.relative_to(base))
        capture = "/runtime/" + str(captures.relative_to(base))
        trigger = "/runtime/" + str(triggers.relative_to(base))
        script = "set -e; source /usr/local/Ascend/cann/set_env.sh; export OMP_NUM_THREADS=1; export PYTHONPATH=/tools:/pair/tools:${PYTHONPATH:-}; "
        script += f"python3 /tools/replay-g0-wy.py --pair /pair --wy /wy --captures {capture} --triggers {trigger} --out {output}/wy; "
        script += f"python3 /tools/check-wy-queued-fixed-inputs.py --pair /pair --wy /wy --captures {capture} --triggers {trigger} --out {output}/queued-repeat.json; "
        for device in (0, 1):
            script += f"python3 /tools/check-wy4-gdn.py --pair /pair --wy /wy --device {device} --out {output}/synthetic-d{device}.jsonl; "
        script += f"python3 /tools/replay-guarded-boundaries.py --pair /pair --wy /wy --captures /runtime/wy-boundary-20261106/capture --selection /runtime/wy-guarded-20261106/boundary-selection.json --out {output}/boundaries; "
        command[-1] = script
        write(out / "launch-command.json", command)
        state("device_running", container=container)
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        code = subprocess.check_output(["podman", "wait", container], text=True, timeout=850).strip()
        with (out / "device.log").open("x") as stream:
            subprocess.run(["podman", "logs", container], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
        assert code == "0", code
        state("device_complete_requires_cpu_reference")
    except BaseException as exc:
        error = repr(exc)
        state("failed", error=error)
    finally:
        signal.alarm(0)
        if container:
            detail = subprocess.run(["podman", "inspect", container], capture_output=True, text=True, timeout=30)
            if detail.returncode == 0:
                inspected = json.loads(detail.stdout)
                if inspected[0]["State"]["Running"]:
                    subprocess.run(["podman", "stop", "--time", "30", container], check=True, timeout=60)
                    inspected = json.loads(subprocess.check_output(["podman", "inspect", container], text=True, timeout=30))
                write(out / "container-inspect.json", inspected)
        after = snapshot()
        clean = after["rootless"]["stdout"] == after["rootful"]["stdout"] == "" and after["primary"]["stdout"] == after["image"]["stdout"] == "inactive"
        write(out / "window-closed.json", {"start_unix_ns": started, "end_unix_ns": time.time_ns(), "elapsed_s": time.time_ns() / 1e9 - started / 1e9, "cleanup_passed": clean, "snapshot": after, "error": error})
        assert clean

if error:
    raise RuntimeError(error)
