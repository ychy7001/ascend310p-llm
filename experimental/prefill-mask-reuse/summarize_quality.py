# SPDX-License-Identifier: Apache-2.0
"""Summarize a 310P same-call audit and first greedy token differences."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import sys


LABEL = re.compile(r'mq-w(\d+)-(fresh|reuse|audit)-(warm-followup|cold-long|cold-short|warm-extraction|prime)')
RANK = re.compile(r'Worker_TP(\d+)')


def step_identity(step):
    requests = {}
    window = None
    for request, count in step['counts']:
        match = LABEL.search(request)
        if not match:
            return None
        current, mode, label = match.groups()
        if window is None:
            window = int(current)
        assert window == int(current) and mode == step['mode'], step
        assert label not in {item[0] for item in requests.values()}
        requests[request] = (label, count)
    ordered = tuple(requests[request] for request in step['order'] if request in requests)
    assert len(ordered) == len(requests), step
    return window, ordered


def first_token_difference(a, b):
    for index, (left, right) in enumerate(zip(a, b, strict=True)):
        if left['token_id'] == right['token_id']:
            continue
        ids = str(left['token_id']), str(right['token_id'])
        return dict(index_zero_based=index, left_token=left['token_id'], right_token=right['token_id'],
                    left_logprobs={i:left['top'].get(i) for i in ids},
                    right_logprobs={i:right['top'].get(i) for i in ids})
    return None


def max_common_prefix_logprob_delta(a, b, difference):
    end = len(a) if difference is None else difference['index_zero_based']
    return max((abs(a[i]['selected_logprob']-b[i]['selected_logprob']) for i in range(end)), default=None)


def main(root, audit_root=None):
    root = Path(root)
    audit_root = root if audit_root is None else Path(audit_root)
    assert (audit_root/'result-code.txt').read_text().strip() == '0'
    roots = [(root, {0,1,2,3,4})] if root == audit_root else [(root, {0,1,2,3}), (audit_root, {4})]
    if root != audit_root:
        for filename in ('image.txt','entry.sha256','model-metadata.sha256','serve/runtime-cli.txt'):
            assert (root/filename).read_bytes() == (audit_root/filename).read_bytes(), filename
    logs = [(directory/'serve/server.log').read_text() for directory,_ in roots]
    assert all('Graph capturing finished' in log and 'Replaying aclgraph' in log for log in logs)
    assert not any('Same-call SplitFuse mismatch' in log for log in logs)
    audit = defaultdict(Counter)
    step_traces = defaultdict(lambda: defaultdict(list))
    for line, allowed in ((line, allowed) for log, (_, allowed) in zip(logs, roots, strict=True)
                          for line in log.splitlines()):
        if '[mask-quality-step] ' not in line:
            continue
        pre, data = line.split('[mask-quality-step] ', 1)
        step = json.loads(data)
        rank = int(RANK.search(pre)[1])
        identity = step_identity(step)
        if identity is None:
            assert not step['checks']
            continue
        window, ordered = identity
        if window not in allowed:
            continue
        if not step['eager']:
            assert step['checks'] == 0
            continue
        step_traces[window][rank].append(ordered)
        assert step['checks'] == sum(n for _,_,n in step['layouts'])
        if step['checks']:
            assert step['mode'] == 'audit'
            for starts, lengths, n in step['layouts']:
                audit[rank][(tuple(starts), tuple(lengths))] += n
                assert len(starts) == len(lengths)+1
    assert set(audit) == {0, 1} and audit[0] == audit[1]
    assert any(len(lengths) > 1 for _,lengths in audit[0])
    assert any(max(lengths) >= 30000 for _,lengths in audit[0])
    batches = [[json.loads(line) for line in (directory/'serve/quality.jsonl').read_text().splitlines()]
               for directory,_ in roots]
    config = batches[0][0]
    assert all(rows[0] == config for rows in batches)
    rows = [config] + [r for batch, (_,allowed) in zip(batches, roots, strict=True)
                       for r in batch[1:] if r['window'] in allowed]
    groups = defaultdict(dict)
    for row in rows[1:]:
        assert row['kind'] == 'request'
        groups[row['window']][row['label']] = row
    assert set(groups) == set(range(len(config['windows'])))
    assert len(rows) == 1 + len(config['windows'])*len(config['slots'])
    labels = [s['label'] for s in config['slots']]
    for window, group in groups.items():
        assert set(group) == set(labels)
        assert set(step_traces[window]) == {0, 1}
        assert step_traces[window][0] == step_traces[window][1]
        for slot in config['slots']:
            r = group[slot['label']]
            assert r['mode'] == config['windows'][window]
            assert r['cached_tokens'] == (11520 if slot['shared'] else 0)
            assert len(r['positions']) == slot['output']

    comparisons = []
    for left,right in ((0,1),(2,3),(0,2),(1,3),(0,4)):
        sa, sb = step_traces[left][0], step_traces[right][0]
        first_schedule_difference = next((i for i in range(max(len(sa),len(sb)))
                                          if i >= len(sa) or i >= len(sb) or sa[i] != sb[i]), None)
        cases = {}
        for label in labels:
            a,b = groups[left][label], groups[right][label]
            difference = first_token_difference(a['positions'],b['positions'])
            cases[label] = dict(first_token_difference=difference, same_text=a['text']==b['text'],
                                 same_token_and_top_logprobs=a['positions']==b['positions'],
                                 max_common_prefix_selected_logprob_delta=max_common_prefix_logprob_delta(
                                     a['positions'],b['positions'],difference))
        comparisons.append(dict(windows=[left,right], modes=[config['windows'][i] for i in (left,right)],
                                first_traced_eager_schedule_difference=first_schedule_difference,
                                left_step=sa[first_schedule_difference] if first_schedule_difference is not None and first_schedule_difference<len(sa) else None,
                                right_step=sb[first_schedule_difference] if first_schedule_difference is not None and first_schedule_difference<len(sb) else None,
                                cases=cases))
    print(json.dumps(dict(config=config, image=(root/'image.txt').read_text().strip(),
        graph_replay=True, same_call_bitwise_equal=True,
        same_call_per_rank=[dict(rank=rank, checks=audit[rank].total(), mixed_checks=sum(n for (_,c),n in audit[rank].items() if len(c)>1),
                                 layouts=[dict(query_starts=a, contexts=b, checks=n) for (a,b),n in audit[rank].items()]) for rank in sorted(audit)],
        model_metadata_sha256=(root/'model-metadata.sha256').read_text().splitlines(),
        http_overlay_sha256=(root/'overlay.sha256').read_text().splitlines(),
        audit_overlay_sha256=(audit_root/'overlay.sha256').read_text().splitlines(),
        runtime_cli=(root/'serve/runtime-cli.txt').read_text().splitlines(),
        traced_eager_steps={window:len(traces[0]) for window,traces in step_traces.items()},
        comparisons=comparisons, http_process_exit_code=int((root/'result-code.txt').read_text()), audit_process_exit_code=0), indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
