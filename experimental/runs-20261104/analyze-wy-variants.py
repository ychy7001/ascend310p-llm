#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
from wy_snapshot_reference import metrics,reference
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--outputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
rows=[]
for path in sorted(a.inputs.glob('rank*-call*.pt')):
 saved=torch.load(path,weights_only=True);actual=torch.load(a.outputs/path.name,weights_only=True);expected=reference(saved['inputs'])
 row={'source':path.name,'metadata':saved['metadata'],'native_vs_cpu':metrics(actual['native_outputs'],expected),'native_vs_torch':metrics(actual['native_outputs'],actual['torch_outputs']),'torch_vs_cpu':metrics(actual['torch_outputs'],expected)}
 rows.append(row)
a.out.write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r['metadata']['variant'],{key:{'pass':all(x['passed'] for x in r[key]),'max_abs':[x['max_abs'] for x in r[key]],'outside':[x['outside_tolerance'] for x in r[key]]} for key in ('native_vs_cpu','native_vs_torch','torch_vs_cpu')})
