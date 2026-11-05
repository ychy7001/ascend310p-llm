#!/usr/bin/env python3
"""Load only the MoE Torch bridge in a fresh process, as the worker does.

Do not preload its C dependency or add its directory to LD_LIBRARY_PATH.
The production SDK environment still supplies the standard Torch/CANN libraries.
Run in the locked runtime window: AscendC metadata loading may consult devices.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def check(library):
    library = Path(library).resolve(strict=True)
    search = [Path(p).resolve() for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p]
    if library.parent in search:
        raise ValueError('Bridge directory must not be supplied through LD_LIBRARY_PATH')
    child = '''import json,sys,torch,torch_npu
from pathlib import Path
dependency_preloaded = any("libmoe_prescale.so" in line
                           for line in Path("/proc/self/maps").read_text().splitlines())
assert not dependency_preloaded
assert "moe_prescale_310::grouped" not in torch._C._dispatch_get_all_op_names()
torch.ops.load_library(sys.argv[1])
assert "moe_prescale_310::grouped" in torch._C._dispatch_get_all_op_names()
print(json.dumps({"cold_bridge_load": True, "dependency_preloaded": dependency_preloaded}))
'''
    result = subprocess.run([sys.executable, '-c', child, str(library)],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError('Cold bridge load failed:\n'+result.stdout+result.stderr)
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence['cold_bridge_load'] and not evidence['dependency_preloaded']
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(check(args.library)), flush=True)


if __name__ == '__main__':
    main()
