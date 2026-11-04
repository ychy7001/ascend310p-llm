# SPDX-License-Identifier: Apache-2.0
"""仅 CPU 读取本实验保存的张量，分析偏差位置与逐位重复性。"""
import argparse
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work',type=Path,default=Path('/work'))
parser.add_argument('--snapshots-dir',type=Path)
a=parser.parse_args()
p=a.snapshots_dir or a.work
saved=torch.load(a.work/'diagnostic-case-1.pt',weights_only=True)
expected=saved['reference']
summary={'scope':'CPU only, no NPU devices exposed; synthetic inputs', 'modes':{}}
for mode in ('torch','native'):
 rows=[];values=[]
 for i in range(12):
  tensors=torch.load(p/f'order-{i}-{mode}.pt',weights_only=True)
  values.append(tensors)
  output=tensors[0].float();delta=(output-expected[0]).abs()
  bad=delta>(.0002+.04*expected[0].abs())
  where=bad.nonzero()
  rows.append(dict(repeat=i,max_abs=delta.max().item(),outside_tolerance=int(bad.sum()),
      affected_tokens=where[:,1].unique().tolist() if len(where) else [],
      affected_heads=where[:,2].unique().tolist() if len(where) else [],
      output_equal_first=bool(torch.equal(tensors[0],values[0][0])),
      state_equal_first=bool(torch.equal(tensors[1],values[0][1]))))
 summary['modes'][mode]=rows
(p/'snapshot-analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
