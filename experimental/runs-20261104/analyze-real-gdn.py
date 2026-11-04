#!/usr/bin/env python3
"""本机 CPU 逐 token 递推验证真实 GDN 快照；不加载 NPU 运行时。"""
import argparse,ast,json
from pathlib import Path
import torch
p=argparse.ArgumentParser()
for name in ('captures','outputs','out','oracle-source'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--candidate',choices=('native','cann'),required=True);a=p.parse_args();torch.set_num_threads(2)
tree=ast.parse(a.oracle_source.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='recurrent_reference')
ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.oracle_source),'exec'),ns)
a.out.mkdir(exist_ok=False);rows=[]
def metrics(actual,expected):
 out=[]
 for name,x,y in zip(('output','state'),actual,expected,strict=True):
  x,y=x.float(),y.float();d=(x-y).abs();bad=d>(.0002+.04*y.abs());rel=((x-y).square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-30)).item()
  finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
  out.append({'output':name,'finite':finite,'max_abs':d.max().item(),'relative_rmse':rel,'outside_tolerance':int(bad.sum()),'elements':y.numel(),'passed':finite and not bool(bad.any()) and rel<=.02})
 return out
for path in sorted(a.captures.glob('rank*-call*.pt')):
 saved=torch.load(path,weights_only=True);cu=saved['metadata']['cu_seqlens'];length=cu[-1];inputs=[x[:,:length].contiguous() for x in saved['inputs']]
 expected=ns['recurrent_reference'](*inputs,saved['initial_state'],cu)
 output=torch.load(a.outputs/path.name,weights_only=True)
 row={'source':path.name,'metadata':saved['metadata'],a.candidate+'_vs_cpu':metrics(output[a.candidate+'_outputs'],expected),'torch_vs_cpu':metrics(output['torch_outputs'],expected),a.candidate+'_vs_torch':metrics(output[a.candidate+'_outputs'],output['torch_outputs'])}
 rows.append(row)
 with (a.out/'rows.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
 print(path.name,{key:all(x['passed'] for x in row[key]) for key in (a.candidate+'_vs_cpu','torch_vs_cpu',a.candidate+'_vs_torch')},flush=True)
summary={key:{'passed':sum(all(x['passed'] for x in r[key]) for r in rows),'total':len(rows),'max_abs':{name:max(r[key][i]['max_abs'] for r in rows) for i,name in enumerate(('output','state'))}} for key in (a.candidate+'_vs_cpu','torch_vs_cpu',a.candidate+'_vs_torch')}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
