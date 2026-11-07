"""Build the isolated K/V=128 library-Cube Gram WY candidate on a2."""
import fcntl
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
record = root / "wy-postbeta-refined-20261107"
candidate_dir = root / "wy-postbeta-refined-v1"


def snapshot():
    commands = {
        "rootless": ["podman", "ps", "-q"],
        "rootful": ["sudo", "-n", "podman", "ps", "-q"],
        "primary": ["systemctl", "is-active", "llm-backend-primary.service"],
        "image": ["systemctl", "--user", "is-active", "qwen36-image.service"],
        "memory": ["free", "-g"],
        "npu": ["npu-smi", "info"],
    }
    return {
        key: {"code": proc.returncode, "stdout": proc.stdout.strip()}
        for key, command in commands.items()
        for proc in [subprocess.run(command, capture_output=True, text=True, timeout=30)]
    }


def state(value, **details):
    (record / "status.json").write_text(json.dumps({"state": value, **details}, indent=2) + "\n")


def build():
    out = candidate_dir
    out.mkdir(exist_ok=False)
    image = "d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a"
    source = root / "native-gdn/source"

    def ignore(path, names):
        skip = {".git", "__pycache__"}
        if Path(path) == source / "csrc":
            skip |= {"build", "build_out", "build_cache", "output"}
        return set(names) & skip

    state("preparing")
    shutil.copytree(source, out / "source", symlinks=True, ignore=ignore)
    target = out / "source/csrc/moe/chunk_gated_delta_rule_compute_wy/op_kernel/arch20/compute_wy_kernel.h"
    original = target.read_bytes()
    expected_blob = "6b2f061fc8d3440bf0942a36c17196181d1d3f8e"
    actual_blob = hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest()
    assert actual_blob == expected_blob, actual_blob

    patcher = root / "patch_pr17640_wy_postbeta_refined.py"
    shutil.copy2(patcher, out / "tools-patcher.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("candidate_patcher", patcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patched = module.patch_text(original)
    target.write_bytes(patched)
    patched_sha = hashlib.sha256(patched).hexdigest()
    assert patched_sha == "1e4b68e09ccae455c2c80415ee4ae50ae906aae37ff8ad05d048af208523009b"

    tools = out / "tools"
    tools.mkdir()
    shutil.copy2(root / "build_pr17640_wy_extension.py", tools)
    template = json.loads((root / "upstream-fwd-o/build-command.json").read_text())
    command = []
    index = 0
    while index < len(template):
        if template[index] == "--device":
            index += 2
            continue
        command.append(template[index])
        index += 1
    command[command.index("--name") + 1] = "postbeta-refined-build-" + str(time.time_ns())
    command[command.index(str(root / "upstream-fwd-o") + ":/work:rw")] = str(out) + ":/work:rw"
    old_image = json.loads((root / "production-manifest.json").read_text())["image"]
    command[command.index(old_image)] = image
    command[-1] = """set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
cd /work/source/csrc
bash build.sh --pkg --ops=chunk_gated_delta_rule_compute_wy --soc=ascend310p --vendor_name=hyperqwen_postbeta_refined -j2
python3 /work/tools/build_pr17640_wy_extension.py --source-root /work/source --build /work/extension
"""
    (out / "source-provenance.json").write_text(
        json.dumps(
            {
                "commit": "18163f6b1d208ab3b4fb06e9752225611c37f40e",
                "image": image,
                "original_kernel_git_blob": actual_blob,
                "patched_kernel_sha256": patched_sha,
                "scope": "Frozen raw-K Gram postbeta candidate plus Neumaier scalar scan only; solve/apply/gates unchanged; no device during build",
            },
            indent=2,
        )
    )
    (out / "build-command.json").write_text(json.dumps(command, indent=2))
    state("building", source_sha256=patched_sha)
    with (out / "build.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    assert result.returncode == 0, result.returncode
    packages = list((out / "source/csrc/build_out").glob("*.run"))
    assert len(packages) == 1, packages
    command[command.index("--name") + 1] = "postbeta-refined-install-" + str(time.time_ns())
    command[-1] = "bash /work/" + str(packages[0].relative_to(out)) + " --quiet --install-path=/work/opp"
    (out / "install-command.json").write_text(json.dumps(command, indent=2))
    with (out / "install.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, result.returncode
    api = out / "opp/vendors/hyperqwen_postbeta_refined_transformer/op_api/lib/libcust_opapi.so"
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(api)], text=True)
    assert "aclnnChunkGatedDeltaRuleComputeWy" in symbols
    artifacts = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in [out / "extension", out / "opp"]
        for path in directory.rglob("*")
        if path.is_file()
    }
    (out / "operator-artifacts.json").write_text(json.dumps(artifacts, indent=2))
    state("built_requires_g0", package=str(packages[0].relative_to(out)), artifacts=len(artifacts))


def interrupted(*_):
    raise TimeoutError("bounded build deadline")


record.mkdir(exist_ok=True)
with open("/tmp/a310-npu.lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    started = time.time_ns()
    error = None
    before = snapshot()
    (record / "build-window-start.json").write_text(json.dumps(before, indent=2))
    assert before["rootless"]["stdout"] == before["rootful"]["stdout"] == ""
    assert before["primary"]["stdout"] == before["image"]["stdout"] == "inactive"
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(3600)
    try:
        build()
    except BaseException as exc:
        error = repr(exc)
        state("failed", error=error)
    finally:
        signal.alarm(0)
        containers = []
        for command_file in [candidate_dir / "build-command.json", candidate_dir / "install-command.json"]:
            if not command_file.exists():
                continue
            command = json.loads(command_file.read_text())
            name = command[command.index("--name") + 1]
            inspected = subprocess.run(["podman", "inspect", name], capture_output=True, text=True, timeout=30)
            if inspected.returncode == 0:
                item = json.loads(inspected.stdout)[0]
                if item["State"]["Running"]:
                    subprocess.run(["podman", "stop", "--time", "30", name], check=True, timeout=60)
                    item = json.loads(subprocess.check_output(["podman", "inspect", name], text=True, timeout=30))[0]
                containers.append(item)
        (record / "build-container-inspects.json").write_text(json.dumps(containers, indent=2))
        after = snapshot()
        clean = (
            after["rootless"]["stdout"] == after["rootful"]["stdout"] == ""
            and after["primary"]["stdout"] == after["image"]["stdout"] == "inactive"
        )
        (record / "build-window-closed.json").write_text(
            json.dumps({"start_unix_ns": started, "end_unix_ns": time.time_ns(), "cleanup_passed": clean, "snapshot": after, "error": error}, indent=2)
        )
        assert clean
    if error:
        raise RuntimeError(error)
