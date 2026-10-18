# SPDX-License-Identifier: Apache-2.0
"""Run pinned community host math tests without importing its serving runtime.
No NPU coverage; package shells only bypass heavy __init__ registration imports.
"""
import argparse,sys,types
from pathlib import Path
import torch,pytest
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--junit',required=True);a=p.parse_args()
for name,relative in [('vllm_ascend','vllm_ascend'),('vllm_ascend.models','vllm_ascend/models'),('vllm_ascend.models.qwen4_exp','vllm_ascend/models/qwen4_exp'),('vllm_ascend.observability','vllm_ascend/observability'),('tests','tests'),('tests.ut','tests/ut'),('tests.ut.qwen38_1m','tests/ut/qwen38_1m')]:
    mod=types.ModuleType(name);mod.__path__=[str(a.source/relative)];sys.modules[name]=mod
sys.path.insert(0,str(a.source));torch.set_num_threads(4)
from vllm_ascend.models.qwen4_exp.qsa_head_sharding import qsa_head_shard
shards=[qsa_head_shard(24,2,256,r,2) for r in range(2)]
assert [(s.query_start,s.num_query_heads,s.kv_start,s.num_kv_heads) for s in shards]==[(0,12,0,1),(12,12,1,1)]
print('TP2 QSA source head placement PASS',shards,flush=True)
names=['test_qsa_attention.py','test_qsa_indexer.py','test_ple_lazy_shard.py']
sys.exit(pytest.main(['-q','--noconftest',f'--junitxml={a.junit}']+[str(a.source/'tests/ut/qwen38_1m'/n) for n in names]))
