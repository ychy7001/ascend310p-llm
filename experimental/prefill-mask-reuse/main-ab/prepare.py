import hashlib
from pathlib import Path
import subprocess
import sys

base, root = map(Path, sys.argv[1:])
rel = Path('vllm_ascend/_310p/attention')
for arm in ('a', 'b'):
    arm_root = root/arm
    dst = arm_root/rel
    dst.mkdir(parents=True, exist_ok=True)
    files = ('attention_v1.py','attention_mask.py')
    for name in files:
        (dst/name).write_bytes((base/rel/name).read_bytes())
    if arm == 'b':
        raw = (root/'candidate.patch').read_text()
        source_diff = raw.split('--- a/tests/', 1)[0]
        subprocess.run(['patch', '--batch', '--forward', '-p1'], input=source_diff, text=True,
                       cwd=arm_root, stdout=sys.stderr, check=True)
        assert 'active_width=active_width' in (dst/'attention_v1.py').read_text()
        assert 'if not active_width:' in (dst/'attention_mask.py').read_text()
    p = dst/'attention_v1.py'
    s = p.read_text()
    begin = s.index('    def forward_chunked_prefill_310(')
    end = s.index('    def forward_impl(', begin)
    part = s[begin:end]
    target = '        if self.support_compressed_mask:\n'
    assert part.count(target)==1
    part = part.replace(target, '''        import os

        compressed = self.support_compressed_mask and os.getenv("MAIN_AB_FORCE_LEGACY") != "1"
        state = attn_metadata.attn_state.name
        if state not in _main_ab_route_seen:
            _main_ab_route_seen.add(state)
            print(f"[main-ab-route] state={state} native_compressed={self.support_compressed_mask} selected_compressed={compressed}", flush=True)
        if compressed:
''')
    # Keep route logging once per implementation instance. Mask shape is audited
    # only on the first multi-token legacy call per worker, outside steady state.
    line = '        torch_npu._npu_paged_attention_splitfuse(\n'
    assert part.count(line)==1
    part=part.replace(line,'''        if state not in _main_ab_mask_seen and (num_actual_tokens >= 128 or state == "SpecDecoding"):
            _main_ab_mask_seen.add(state)
            print(f"[main-ab-mask] shape={tuple(mask.shape)} query={num_actual_tokens} state={state}", flush=True)
'''+line)
    p.write_text(s[:begin]+part+s[end:]+'\n_main_ab_route_seen = set()\n_main_ab_mask_seen = set()\n')
    outputs = [dst / name for name in files]
    for out in outputs:
        compile(out.read_text(),str(out),'exec')
        print(hashlib.sha256(out.read_bytes()).hexdigest(),out)
