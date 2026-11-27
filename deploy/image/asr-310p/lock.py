"""Calculate or verify the exact inputs for the offline ASR build."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parent
lock_path = root.parent / "asr-310p.lock.json"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def inputs():
    image = (root.parent / "Containerfile.asr-310p").read_text().splitlines()[0].split()[1]
    overlay = {}
    wheels = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root)
            (wheels if relative.parts[0] == "wheels" else overlay)[relative.as_posix()] = digest(path)
    listed = {
        line.split("--hash=sha256:")[1] for line in (root / "stream-wheels.txt").read_text().splitlines()
        if line and not line.startswith("#")
    }
    if len(wheels) != len(listed) or set(wheels.values()) != listed:
        raise SystemExit("Wheel SHA256s do not match stream-wheels.txt")
    patched = subprocess.run(
        ["patch", "--quiet", "--batch", "--forward", "--fuzz=0", "-o", "-",
         root / "ws_server.py", root / "isolate-stream-vad.patch"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    return {
        "base": {
            "image": image,
            "architecture": "amd64",
            "index_digest": "sha256:041cc7f7b9e9c51cd823da28dc5a6c8ea1ac06df2b203e69a82766fc1668eafa",
        },
        "overlay_sha256": overlay,
        "wheel_manifest": "stream-wheels.txt",
        "wheel_sha256": wheels,
        "vad_patch": {
            "original": "ws_server.py",
            "patch": "isolate-stream-vad.patch",
            "patched_ws_server_sha256": sha256(patched).hexdigest(),
        },
    }


if len(sys.argv) == 2 and sys.argv[1] == "--write":
    lock_path.write_text(json.dumps(inputs(), indent=2) + "\n")
else:
    lock = json.loads(lock_path.read_text())
    if len(sys.argv) == 2:
        if digest(Path(sys.argv[1])) != lock["vad_patch"]["patched_ws_server_sha256"]:
            raise SystemExit("VAD-patched ws_server.py SHA256 mismatch")
        print("VAD-patched ws_server.py SHA256 verified")
    else:
        if inputs() != lock:
            raise SystemExit("ASR image source differs from asr-310p.lock.json")
        print(f"Verified {len(lock['overlay_sha256'])} overlay files, {len(lock['wheel_sha256'])} wheels, "
              "and VAD-patched ws_server.py SHA256")
