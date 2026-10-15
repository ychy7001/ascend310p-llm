"""隔离候选bootstrap：先保留现场所有补丁，再替换实验greedy函数。"""
from pathlib import Path
exec(compile(Path('/c8bootstrap/sitecustomize.py').read_bytes(), '/c8bootstrap/sitecustomize.py', 'exec'))
import bounded_greedy
bounded_greedy.install()
