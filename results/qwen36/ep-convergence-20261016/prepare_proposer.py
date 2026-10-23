"""从已核验 clt 源码生成独立候选，不修改安装目录。"""
import argparse
import ast
import hashlib
import subprocess
from pathlib import Path

EXPECTED = "4a90532c5c3667d9eb4fec666bf03c65a2f1c92cd402f731b644ef8fa2ed82ab"
p = argparse.ArgumentParser()
p.add_argument("original", type=Path)
p.add_argument("output", type=Path)
args = p.parse_args()
data = args.original.read_bytes()
if hashlib.sha256(data).hexdigest() != EXPECTED:
    raise SystemExit("原始源码 SHA256 不匹配，拒绝应用固定版本补丁")
if args.output.exists():
    raise SystemExit("输出已存在，拒绝覆盖")
patch = Path(__file__).with_name("slot-device.patch")
result = subprocess.run(["patch", "--silent", "--output", str(args.output),
                         str(args.original), str(patch)], check=True)
ast.parse(args.output.read_text())
print(hashlib.sha256(args.output.read_bytes()).hexdigest())
