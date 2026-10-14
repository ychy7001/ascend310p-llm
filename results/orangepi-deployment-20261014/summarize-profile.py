"""汇总已导出的设备任务；排除事件等待，不能用重叠任务求和代替墙钟。"""
import argparse
import collections
import csv
import gzip
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
a = p.parse_args()
files = list(a.directory.glob('task_time_*.csv')) + list(a.directory.glob('task_time_*.csv.gz'))
assert len(files) == 1, 'Expected exactly one task_time CSV'
opener = gzip.open if files[0].suffix == '.gz' else open
with opener(files[0], 'rt', newline='') as stream:
    rows = list(csv.DictReader(stream))
groups = collections.defaultdict(lambda: {'count': 0, 'duration_us': 0.0})
types = collections.defaultdict(float)
for row in rows:
    duration = float(row['task_time(us)'])
    types[row['kernel_type']] += duration
    if row['kernel_type'] not in ('AI_CORE', 'AI_VECTOR_CORE', 'AI_CPU', 'MEMCPY_ASYNC'):
        continue
    item = groups[row['kernel_name']]
    item['count'] += 1
    item['duration_us'] += duration
span = max(float(r['task_stop(us)']) for r in rows) - min(float(r['task_start(us)']) for r in rows)
print(json.dumps({
    'source': files[0].name,
    'scope': __doc__,
    'device_task_span_us': span,
    'duration_by_type_us': dict(types),
    'kernels': sorted(({'name': k, **v} for k, v in groups.items()),
                      key=lambda x: x['duration_us'], reverse=True),
}, ensure_ascii=False, indent=2))
