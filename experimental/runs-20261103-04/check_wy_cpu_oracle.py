# SPDX-License-Identifier: Apache-2.0
"""仅 CPU：用生产 chunk 参考验证已保存的111-token递推参考。"""
import ast
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
p=Path('/work')
source=p/'production-chunk.py'
tree=ast.parse(source.read_text())
functions=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)],type_ignores=[])
ns=dict(torch=torch,F=torch.nn.functional,CHUNK_SIZE=64)
exec(compile(functions,str(source),'exec'),ns)
saved=torch.load(p/'diagnostic-case-1.pt',weights_only=True)
q,k,v,g,beta,state=saved['cpu_inputs']
chunk=ns['chunk_gated_delta_rule_pytorch'](q,k,v,g,beta,initial_state=state,output_final_state=True)
rows=[]
for a,b in zip(chunk,saved['reference']):
    torch.testing.assert_close(a.float(),b.float(),atol=.00001,rtol=.001)
    rows.append(dict(max_abs=(a.float()-b.float()).abs().max().item(),relative_rmse=((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt()).item()))
report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tokens=111,hk=8,hv=16,nonzero_state=True,
            cpu_chunk_vs_token_recurrence=rows,status='PASS',scope='CPU-only independent oracle cross-check on exact saved failure input')
(p/'cpu-oracle-111.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
