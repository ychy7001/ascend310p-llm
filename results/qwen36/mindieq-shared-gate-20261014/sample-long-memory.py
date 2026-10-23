# SPDX-License-Identifier: Apache-2.0
"""附加观测：指定长测试进程存活期间记录npu-smi，每20秒一帧，不执行推理。"""
import argparse,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid',type=int,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();proc=Path(f'/proc/{a.pid}/cmdline')
with a.out.open('x') as output:
 while proc.exists():
  try:command=proc.read_bytes().decode().replace('\0',' ')
  except FileNotFoundError:break
  if 'bench_server_phases.py' not in command or '--prompt-tokens 114560' not in command:break
  result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15)
  output.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})+'\n');output.flush()
  time.sleep(20)
