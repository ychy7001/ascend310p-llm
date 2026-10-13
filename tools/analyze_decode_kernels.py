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
opener = gzip.open if a.input.suffix == '.gz' else open
with opener(a.input, 'rt') as f:
    for r in csv.DictReader(f):
        rows += 1
        duration = float(r['Duration(us)'])
        groups[r['Name']][0] += 1
        groups[r['Name']][1] += duration
        key = (r['Name'], r['Input Shapes'], r['Input Data Types'])
        shapes[key][0] += 1
        shapes[key][1] += duration
assert rows > 0
summed = sum(v[1] for v in groups.values())
summary = {'scope': __doc__, 'kernel_count': rows, 'sum_kernel_us': summed,
           'top': [{'name': n, 'count': v[0], 'sum_us': v[1], 'sum_fraction': v[1] / summed}
                   for n, v in sorted(groups.items(), key=lambda item: -item[1][1])[:30]],
           'top_shapes': [{'name': k[0], 'shapes': k[1], 'dtypes': k[2], 'count': v[0], 'sum_us': v[1]}
                          for k, v in sorted(shapes.items(), key=lambda item: -item[1][1])[:30]]}
if a.reference:
    reference = json.loads(a.reference.read_text())
    old = {r['name']: r for r in reference['top']}
    summary['reference_comparison'] = [dict(r, reference_sum_fraction=old[r['name']]['sum_fraction'],
        reference_mean_us=old[r['name']]['sum_us'] / old[r['name']]['count'], mean_us=r['sum_us'] / r['count'])
        for r in summary['top'] if r['name'] in old]
a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'kernel_count': rows, 'sum_kernel_us': summed}))
