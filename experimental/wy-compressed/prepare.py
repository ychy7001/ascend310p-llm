#!/usr/bin/env python3
"""Add one-shot route markers to the retained 0.30 WY image sources."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('out', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gdn = (args.source / 'ops/fla/chunk_gated_delta_rule.py').read_text()
    start = '    return torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(\n'
    assert gdn.count(start) == 1
    a, b = gdn.split(start)
    end = '        chunk_size,\n    )\n'
    call, rest = b.split(end, 1)
    gdn = a + start.replace('return ', 'result = ') + call + end + (
        '    global _wy_compressed_probe_seen\n'
        '    if not _wy_compressed_probe_seen and q.shape[1] >= 1024:\n'
        '        print("[wy-compressed-probe] native WY invoked", flush=True)\n'
        '        _wy_compressed_probe_seen = True\n'
        '    return result\n'
    ) + rest + '\n_wy_compressed_probe_seen = False\n'
    (args.out / 'chunk_gated_delta_rule.py').write_text(gdn)
    attn = (args.source / 'attention/attention_v1.py').read_text()
    start = '            torch_npu._npu_paged_attention_splitfuse_v2(\n'
    assert attn.count(start) == 1
    a, b = attn.split(start)
    end = '            )\n'
    call, rest = b.split(end, 1)
    attn = a + start + call + end + (
        '            global _splitfuse_probe_seen\n'
        '            if not _splitfuse_probe_seen and num_actual_tokens >= 1024:\n'
        '                print("[wy-compressed-probe] compressed SplitFuse invoked", flush=True)\n'
        '                _splitfuse_probe_seen = True\n'
    ) + rest + '\n_splitfuse_probe_seen = False\n'
    (args.out / 'attention_v1.py').write_text(attn)


if __name__ == '__main__':
    main()
