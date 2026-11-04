#!/usr/bin/env python3
"""按首次真实误差坐标裁剪输入，并生成有界单变量对照；只在 CPU 执行。"""
import argparse,json
from pathlib import Path
import torch
from wy_snapshot_reference import reference
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);a.out.mkdir(exist_ok=False)
saved=torch.load(a.capture,weights_only=True);replay=torch.load(a.replay,weights_only=True)
inputs=saved['inputs'];cpu=reference(inputs)
ratio=inputs[2].shape[2]//inputs[0].shape[2]
error=(replay['native_outputs'][3].float()-cpu[3]).abs()/(.0005+.04*cpu[3].abs())
coord=torch.unravel_index(error.argmax(),error.shape);b,h,t,d=map(int,coord)
start=t//64*64;kh=h//ratio;hvstart=kh*ratio
variants={'full':inputs}
variants['chunk_all_heads']=[x[:,start:start+64].contiguous() for x in inputs]
variants['chunk_head_pair']=[x[:,start:start+64,kh:kh+1].contiguous() if i<2 else x[:,start:start+64,hvstart:hvstart+ratio].contiguous() for i,x in enumerate(inputs)]
variants['chunk_single_head']=[x[:,start:start+64,kh:kh+1].contiguous() if i<2 else x[:,start:start+64,h:h+1].contiguous() for i,x in enumerate(inputs)]
base=variants['chunk_head_pair']
for scale in (.1,.01):
 values=[x.clone() for x in base];values[2]=(values[2].float()*scale).half()
 variants[f'v_scale_{scale}']=values
values=[x.clone() for x in base];values[3].fill_(-1);variants['g_constant_minus1']=values
values=[x.clone() for x in base];torch.manual_seed(17640);values[1]=torch.nn.functional.normalize(torch.randn_like(values[1]).float(),dim=-1).half();variants['k_random_normalized']=values
values=[x.clone() for x in base];values[4].fill_(.5);variants['beta_constant_half']=values
for index,(name,values) in enumerate(variants.items()):
 meta={'rank':0,'call':index+1,'layer':saved['metadata']['layer'],'variant':name,'source':a.capture.name,'source_coordinate':[b,h,t,d],'chunk_start':start,
       'original_cu_seqlens':saved['metadata']['cu_seqlens'],'initial_state_required_by_WY':False,
       'inputs':[{'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype)} for x in values]}
 torch.save({'metadata':meta,'inputs':values,'torch_outputs':[], 'initial_state':None,'cu_seqlens':None},a.out/f'rank0-call{index+1:04d}.pt')
(a.out/'variants.json').write_text(json.dumps({'source_coordinate':[b,h,t,d],'error_over_tolerance':error.max().item(),'variants':list(variants)},indent=2)+'\n')
print((a.out/'variants.json').read_text())
