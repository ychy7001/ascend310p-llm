#!/usr/bin/env python3
"""本机 CPU 分析真实 WY 输入和离线输出；只读快照，不导入 NPU。"""
import argparse
import json
from pathlib import Path
import torch
from wy_snapshot_reference import reference, metrics
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--captures',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--replay',type=Path)
p.add_argument('--candidate',choices=('native','cann'))
a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(exist_ok=True)
rows=[]
for path in sorted(a.captures.glob('rank*-call*.pt')):
 saved=torch.load(path,weights_only=True)
 inputs=saved['inputs'];expected=reference(inputs)
 row={'source':path.name,'metadata':saved['metadata'],
      'captured_torch_vs_cpu':metrics(saved['torch_outputs'],expected)}
 if a.replay:
  replay=torch.load(a.replay/path.name,weights_only=True)
  row.update({a.candidate+'_vs_cpu':metrics(replay[a.candidate+'_outputs'],expected),
              a.candidate+'_vs_torch':metrics(replay[a.candidate+'_outputs'],replay['torch_outputs']),
              'replayed_torch_vs_cpu':metrics(replay['torch_outputs'],expected)})
 q,k,v,g,beta=inputs
 row['input_ranges']={name:{'min':x.min().item(),'max':x.max().item(),'rms':x.float().square().mean().sqrt().item()} for name,x in zip(('q','k','v','g','beta'),inputs)}
 row['k_l2_norm_max']=k.float().norm(dim=-1).max().item()
 row['g_cumulative_min']=expected[4].min().item()
 rows.append(row)
 with (a.out/'rows.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
 print(path.name, 'cpu torch pass',all(x['passed'] for x in row['captured_torch_vs_cpu']),
       'candidate pass',all(x['passed'] for x in row.get((a.candidate or '')+'_vs_cpu',[])),flush=True)
summary={'calls':len(rows),'captured_torch_vs_cpu_pass':sum(all(x['passed'] for x in r['captured_torch_vs_cpu']) for r in rows)}
if a.replay:
 for comparison in (a.candidate+'_vs_cpu',a.candidate+'_vs_torch','replayed_torch_vs_cpu'):
  summary[comparison]={'passed':sum(all(x['passed'] for x in r[comparison]) for r in rows),
   'first_failure_by_rank':{str(rank):next(({'source':r['source'],'metadata':r['metadata'],'metrics':r[comparison]} for r in rows if r['metadata']['rank']==rank and not all(x['passed'] for x in r[comparison])),None) for rank in (0,1)},
   'max_abs_by_output':{name:max(r[comparison][i]['max_abs'] for r in rows) for i,name in enumerate(('q','k','w','u','g'))}}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2),flush=True)
