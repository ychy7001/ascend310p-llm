# SPDX-License-Identifier: Apache-2.0
"""仅CPU验证两种W4布局的字节关系；不声称设备实现完成。"""
import argparse,ast,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream-test',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
mod=ast.parse(a.upstream_test.read_text());fn=next(x for x in mod.body if isinstance(x,ast.FunctionDef) and x.name=='pack_biased_nz_int32_vec');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.upstream_test),'exec'),ns)
rng=np.random.default_rng(20261013);rows=[]
for n,k in [(1024,2048),(2048,512),(16,128),(32,256)]:
 w=rng.integers(-8,8,(n,k),dtype=np.int8)
 ck=((w[:,0::2]&15)|((w[:,1::2]&15)<<4)).astype(np.uint8)
 native=ck.reshape(n//16,16,k//128,2,32).transpose(0,2,3,1,4).copy().reshape(n//16,k//64,16,32)
 mind=native.transpose(1,0,2,3).copy().reshape(-1)^np.uint8(0x88)
 ref=ns['pack_biased_nz_int32_vec'](w.T.copy(),k,n).view(np.uint8)
 assert np.array_equal(mind,ref),(n,k)
 rows.append({'n':n,'k':k,'status':'PASS','weight_bytes':mind.nbytes})
with a.out.open('x') as f:json.dump({'scope':__doc__,'seed':20261013,'cases':rows},f,indent=2)
print(json.dumps(rows))
