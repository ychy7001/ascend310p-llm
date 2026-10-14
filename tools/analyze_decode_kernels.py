# SPDX-License-Identifier: Apache-2.0
"""按捕获kernel耗时求和，不等于墙钟比例；可读取gzip原始导出。"""
import argparse
import collections
import csv
import gzip
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--reference', type=Path)
a = p.parse_args()
groups = collections.defaultdict(lambda: [0, 0.0])
shapes = collections.defaultdict(lambda: [0, 0.0])
rows = 0
starts, stops = [], []
opener = gzip.open if a.input.suffix == '.gz' else open
with opener(a.input, 'rt') as f:
    reader = csv.DictReader(f)
    # torch-npu kernel_details and msprof op_summary use different headers.
    name_col = 'Name' if 'Name' in reader.fieldnames else 'Op Name'
    duration_col = 'Duration(us)' if 'Duration(us)' in reader.fieldnames else 'Task Duration(us)'
    for r in reader:
        rows += 1
        duration = float(r[duration_col])
        groups[r[name_col]][0] += 1
        groups[r[name_col]][1] += duration
        key = (r[name_col], r['Input Shapes'], r['Input Data Types'])
        shapes[key][0] += 1
        shapes[key][1] += duration
        if 'Task Start Time(us)' in r:
            start = float(r['Task Start Time(us)'])
            starts.append(start)
            stops.append(start + duration)
assert rows > 0
summed = sum(v[1] for v in groups.values())
summary = {'scope': __doc__, 'kernel_count': rows, 'sum_kernel_us': summed,
           'top': [{'name': n, 'count': v[0], 'sum_us': v[1], 'sum_fraction': v[1] / summed}
                   for n, v in sorted(groups.items(), key=lambda item: -item[1][1])[:30]],
           'top_shapes': [{'name': k[0], 'shapes': k[1], 'dtypes': k[2], 'count': v[0], 'sum_us': v[1]}
                          for k, v in sorted(shapes.items(), key=lambda item: -item[1][1])[:30]]}
if starts:
    summary['device_task_span_us'] = max(stops) - min(starts)
    summary['span_warning'] = 'Device span may cover only a bounded capture window; validate against HTTP scope. Concurrent task durations overlap.'
if a.reference:
    reference = json.loads(a.reference.read_text())
    old = {r['name']: r for r in reference['top']}
    summary['reference_comparison'] = [dict(r, reference_sum_fraction=old[r['name']]['sum_fraction'],
        reference_mean_us=old[r['name']]['sum_us'] / old[r['name']]['count'], mean_us=r['sum_us'] / r['count'])
        for r in summary['top'] if r['name'] in old]
a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'kernel_count': rows, 'sum_kernel_us': summed}))
