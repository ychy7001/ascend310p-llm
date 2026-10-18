# SPDX-License-Identifier: Apache-2.0
"""Sample host memory and raw npu-smi output; sampled maxima are not exact peaks."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=5)
    args = parser.parse_args()
    if args.interval <= 0 or args.seconds <= 0:
        parser.error('Durations must be positive')
    deadline = time.monotonic() + args.seconds
    with args.output.open('a') as out:
        while time.monotonic() < deadline:
            memory = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key, value = line.split(':', 1)
                if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                    memory[key + '_KiB'] = int(value.split()[0])
            result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
            out.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                  'host_memory': memory, 'npu_smi_exit_code': result.returncode,
                                  'npu_smi': result.stdout, 'stderr': result.stderr}) + '\n')
            out.flush()
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
