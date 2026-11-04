import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
s=torch.load(a.input,weights_only=True);q,k,v,g,beta=s['inputs']
key=k.float()[0,:,0];gram=key@key.T;lower=gram.tril(-1);gate=g[0,:,0];cum=gate.cumsum(0)
A=(beta[0,:,0,None].float()*gram*torch.exp((cum[:,None]-cum[None,:]).tril()).tril()).tril(-1)
print(json.dumps({'shapes':[list(x.shape) for x in s['inputs']], 'ranges':{n:{'min':x.min().item(),'max':x.max().item(),'rms':x.float().square().mean().sqrt().item()} for n,x in zip(('q','k','v','g','beta'),s['inputs'])},'k_offdiag_abs_max':lower.abs().max().item(),'k_adjacent_cosine_mean':torch.nn.functional.cosine_similarity(key[1:],key[:-1]).mean().item(),'g_chunk_sum':gate.sum().item(),'A_max_absolute_row_sum':A.abs().sum(-1).max().item(),'A_frobenius_squared':A.square().sum().item()},indent=2))
