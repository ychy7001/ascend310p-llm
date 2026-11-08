"""Experimental: use int64 slot-offset arithmetic in eager MTP preparation."""
import hashlib
import json
from pathlib import Path

p = Path('/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py')
old = p.read_text()
needle = '            block_ids = block_ids.view(-1)\n'
assert old.count(needle) == 1, 'unexpected source; do not patch'
new = old.replace(needle, '            block_ids = block_ids.view(-1).to(torch.int64)\n')
p.write_text(new)
print(json.dumps({'file': str(p), 'before_sha256': hashlib.sha256(old.encode()).hexdigest(),
                  'after_sha256': hashlib.sha256(new.encode()).hexdigest(),
                  'change': 'slot block IDs promoted to int64 before block-size multiplication'}))
