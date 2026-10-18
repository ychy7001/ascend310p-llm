# SPDX-License-Identifier: Apache-2.0
"""Conservative weight-only TP2 budget from pinned HF index/file sizes.
Does not count workspace, graph, KV, allocator fragmentation or load peaks.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--blobs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
wm=json.loads(a.index.read_text())['weight_map'];blobs=json.loads(a.blobs.read_text());sizes={x['rfilename']:x['size'] for x in blobs['siblings']}
files=defaultdict(set)
for name,file in wm.items():
 category='experts' if file.startswith('experts-') else 'ple' if '.ple_embedding.ngram_embedding.shard_' in name else 'mtp' if name.startswith('mtp.') else 'vision' if name.startswith('model.visual.') else 'base_float'
 files[category].add(file)
for c,fs in files.items():
 for d,other in files.items():
  if c!=d:assert not fs.intersection(other),(c,d)
# 48 layers, 512 experts, 3 projections, 640*2560 coefficients each.
params=48*512*3*640*2560;groups=params//128
native=params//2+groups*6 # scale+offset+sum, all fp16
base=sum(sizes[f] for f in files['base_float'])
embed=2*248320*2560*2 # embedding and independent LM head
result={'revision':blobs['sha'],'category_files':{c:len(fs) for c,fs in files.items()},'category_file_bytes':{c:sum(sizes[f] for f in fs) for c,fs in files.items()},'native_expert_storage_bytes':native,'native_expert_storage_per_tp2_rank_bytes':native//2,'base_float_conservative_replicated_bytes':base,'tp2_rank_weight_upper_bound_all_base_replicated_bytes':native//2+base,'tp2_rank_weight_upper_bound_only_vocab_parallel_bytes':native//2+base-embed//2,'source_supported_additional_sharding':'GDN/QSA projections, shared experts; not included in conservative bound','not_counted':['KV/QSA/indexer caches','GDN/PLE states','workspace','graph pools','allocator fragmentation','load transient','host page cache'],'scope':'text only, MTP=0; file headers included in float bound, experts use actual native runtime layout; TP2/EP2 source ownership only, not allocation measurement'}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:(v/2**30 if k.endswith('bytes') and isinstance(v,int) else v) for k,v in result.items()},indent=2))
