"""独立EP候选，保留现场bootstrap，仅阻塞上传greedy计数。"""
from pathlib import Path
exec(compile(Path('/c8bootstrap/sitecustomize.py').read_bytes(), '/c8bootstrap/sitecustomize.py', 'exec'))
import speculative_metadata_310
speculative_metadata_310.install()
