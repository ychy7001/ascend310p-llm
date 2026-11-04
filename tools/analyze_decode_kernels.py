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
intervals = []
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
        timestamp = r.get('Task Start Time(us)') or r.get('Start Time(us)')
        if timestamp is not None:
            start = float(timestamp)
            starts.append(start)
            stops.append(start + duration)
            intervals.append((start, start + duration))
assert rows > 0
summed = sum(v[1] for v in groups.values())
summary = {'scope': __doc__, 'kernel_count': rows, 'sum_kernel_us': summed,
           'top': [{'name': n, 'count': v[0], 'sum_us': v[1], 'sum_fraction': v[1] / summed}
                   for n, v in sorted(groups.items(), key=lambda item: -item[1][1])[:30]],
           'top_shapes': [{'name': k[0], 'shapes': k[1], 'dtypes': k[2], 'count': v[0], 'sum_us': v[1]}
                          for k, v in sorted(shapes.items(), key=lambda item: -item[1][1])[:30]]}
if starts:
    summary['device_task_span_us'] = max(stops) - min(starts)
    # 跨stream取区间并集，避免把重叠kernel耗时求和当成设备占用率。
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    active = sum(stop - start for start, stop in merged)
    span = summary['device_task_span_us']
    summary['captured_kernel_interval_union_us'] = active
    summary['captured_kernel_interval_coverage'] = active / span if span > 0 else None
    summary['captured_span_without_recorded_kernel_us'] = span - active
    summary['span_warning'] = 'Device span may cover only a bounded capture window; validate against HTTP scope. Concurrent task durations overlap.'
if a.reference:
    reference = json.loads(a.reference.read_text())
    old = {r['name']: r for r in reference['top']}
    summary['reference_comparison'] = [dict(r, reference_sum_fraction=old[r['name']]['sum_fraction'],
        reference_mean_us=old[r['name']]['sum_us'] / old[r['name']]['count'], mean_us=r['sum_us'] / r['count'])
        for r in summary['top'] if r['name'] in old]
a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'kernel_count': rows, 'sum_kernel_us': summed}))
