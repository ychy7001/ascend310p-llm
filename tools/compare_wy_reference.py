#!/usr/bin/env python3
"""Adjudicate the frozen five-arm CPU-subset WY reference protocol.

Missing top-k targets produce intervals; unavailable metrics never pass.
Conditional KL uses the *same* five-way top-20 intersection at each position.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ('R', 'A', 'A_prime', 'B', 'B0')


def target_distance(left, right):
    x, y = left['target_logprob'], right['target_logprob']
    if x is not None and y is not None:
        d = abs(x - y)
        return d, d
    if x is None and y is None:
        return 0.0, math.inf
    exact, bound = (y, left['target_upper_bound']) if x is None else (x, right['target_upper_bound'])
    if bound is None:
        raise ValueError('Censored target needs an explicit upper bound')
    return max(0.0, exact - bound), math.inf


def judge(candidate, baseline, repeat):
    """Conservative interval proof of B-A <= abs(A_prime-A)."""
    if any(value is None for value in (candidate, baseline, repeat)):
        return {'passed': None, 'reason': 'unavailable metric'}
    blo, bhi = candidate; alo, ahi = baseline; plo, phi = repeat
    difference = blo - ahi, bhi - alo
    noise = max(0.0, plo - ahi, alo - phi), max(phi - alo, ahi - plo)
    passed = True if difference[1] <= noise[0] else False if difference[0] > noise[1] else None
    def finite(value):
        return value if math.isfinite(value) else ('+infinity' if value > 0 else '-infinity')
    return {'passed': passed, 'candidate_loss': list(map(finite, candidate)),
            'A_loss': list(map(finite, baseline)), 'A_prime_loss': list(map(finite, repeat)),
            'B_minus_A': list(map(finite, difference)), 'noise': list(map(finite, noise))}


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def analyze(records):
    if set(records) != set(ARMS):
        raise ValueError('Expected exactly R/A/A_prime/B/B0')
    reference = records['R']
    for arm in ARMS:
        if records[arm].get('diagnostic_only'):
            raise ValueError('Diagnostic outputs cannot establish the R gate')
        if records[arm]['input_sha256'] != reference['input_sha256'] or records[arm]['continuation_sha256'] != reference['continuation_sha256']:
            raise ValueError('Input or frozen continuation hash mismatch')
    by_arm = {arm: {c['id']: c for c in records[arm]['cases']} for arm in ARMS}
    expected_ids = set(by_arm['R'])
    if any(set(by_arm[arm]) != expected_ids for arm in ARMS) or any(len(by_arm[arm]) != len(records[arm]['cases']) for arm in ARMS):
        raise ValueError('Case IDs differ or are duplicated')
    losses = {arm: {'top1_disagreement_rate': [], 'mean_conditional_topk_kl': [],
                    'max_absolute_target_logprob_difference': [], 'first_divergence': {}}
              for arm in ARMS if arm != 'R'}
    supports = []; censored = []; position_count = 0
    for case in reference['cases']:
        cid = case['id']; count = len(case['positions'])
        if not count:
            raise ValueError('Empty reference case')
        if any(len(by_arm[arm][cid]['positions']) != count for arm in ARMS):
            raise ValueError('Position counts differ')
        first = {arm: None for arm in losses}
        for pos in range(count):
            rows = {arm: by_arm[arm][cid]['positions'][pos] for arm in ARMS}
            r = rows['R']; position_count += 1
            for arm, row in rows.items():
                if row['position'] != pos or row['context_sha256'] != r['context_sha256'] or row['target'] != r['target']:
                    raise ValueError('Teacher-forced context, target or position mismatch')
                probs = row['logprobs']
                if len(probs) != 20 or not all(math.isfinite(v) and v <= 1e-6 for v in probs.values()):
                    raise ValueError('Expected 20 finite normalized logprobs')
                if str(row['top1']) not in probs or probs[str(row['top1'])] != max(probs.values()):
                    raise ValueError('top1 is not an argmax of returned logprobs')
                if row['target_logprob'] is None:
                    if str(row['target']) in probs or row['target_upper_bound'] != min(probs.values()):
                        raise ValueError('Invalid target censoring bound')
                    censored.append({'case': cid, 'position': pos, 'arm': arm, 'upper_bound': row['target_upper_bound']})
                elif not math.isfinite(row['target_logprob']) or row['target_logprob'] > 1e-6:
                    raise ValueError('Nonfinite or positive target logprob')
                elif str(row['target']) in probs and row['target_logprob'] != probs[str(row['target'])]:
                    raise ValueError('Explicit target disagrees with returned top20')
            support = sorted(set.intersection(*(set(rows[arm]['logprobs']) for arm in ARMS)))
            masses = {arm: math.fsum(math.exp(rows[arm]['logprobs'][k]) for k in support) for arm in ARMS}
            supports.append({'case': cid, 'position': pos, 'size': len(support), 'covered_mass': masses})
            normalizers = {arm: logsumexp([rows[arm]['logprobs'][k] for k in support]) for arm in ARMS} if support else None
            for arm in losses:
                mismatch = int(rows[arm]['top1'] != r['top1'])
                losses[arm]['top1_disagreement_rate'].append(float(mismatch))
                if mismatch and first[arm] is None:
                    first[arm] = pos
                kl = None
                if support:
                    kl = math.fsum(math.exp(r['logprobs'][k] - normalizers['R']) *
                        ((r['logprobs'][k] - normalizers['R']) - (rows[arm]['logprobs'][k] - normalizers[arm])) for k in support)
                    # Roundoff at zero is not a probability floor or gate allowance.
                    kl = max(0.0, kl)
                losses[arm]['mean_conditional_topk_kl'].append(kl)
                losses[arm]['max_absolute_target_logprob_difference'].append(target_distance(rows[arm], r))
        for arm in losses:
            losses[arm]['first_divergence'][cid] = {'index': first[arm],
                'loss': (count - first[arm]) / count if first[arm] is not None else 0.0}
    aggregates = {}
    for arm, values in losses.items():
        top1 = math.fsum(values['top1_disagreement_rate']) / position_count
        kl = values['mean_conditional_topk_kl']
        targets = values['max_absolute_target_logprob_difference']
        aggregates[arm] = {'top1_disagreement_rate': (top1, top1),
            'mean_conditional_topk_kl': None if any(v is None for v in kl) else (math.fsum(kl) / position_count,) * 2,
            'max_conditional_topk_kl': None if any(v is None for v in kl) else (max(kl),) * 2,
            'max_absolute_target_logprob_difference': (max(v[0] for v in targets), max(v[1] for v in targets))}
        aggregates[arm].update({'first_divergence/' + cid: (v['loss'], v['loss']) for cid, v in values['first_divergence'].items()})
    checks = {metric: judge(aggregates['B'][metric], aggregates['A'][metric], aggregates['A_prime'][metric]) for metric in aggregates['B']}
    flags = [c['passed'] for c in checks.values()]
    verdict = False if False in flags else None if None in flags else True
    # JSON has no Infinity; expose bounds explicitly rather than nonstandard JSON.
    safe_aggregates = {arm: {key: None if value is None else [v if math.isfinite(v) else '+infinity' for v in value]
                            for key, value in metrics.items()} for arm, metrics in aggregates.items()}
    return {'passed': verdict, 'verdict': 'pass' if verdict is True else 'fail' if verdict is False else 'indeterminate',
            'cases': len(reference['cases']), 'positions': position_count, 'checks': checks,
            'losses_vs_R': safe_aggregates, 'support': supports, 'censored_targets': censored,
            'first_divergence': {arm: loss['first_divergence'] for arm, loss in losses.items()},
            'limitations': ['KL is conditional on the five-way top20 intersection, not full-vocabulary KL.',
                            'Interval uncertainty or unavailable metrics cannot pass. B0 is diagnostic only.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        p.add_argument('--' + arm, type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    paths = {arm: getattr(args, arm) for arm in ARMS}
    records = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    selection = json.loads(args.protocol.read_text())['cpu_subset_protocol']
    expected = {c['case']: c['stop_exclusive'] for c in selection['positions']}
    for arm, record in records.items():
        actual = {c['id']: len(c['positions']) for c in record['cases']}
        if actual != expected or record['input_sha256'] != selection['input_sha256']:
            raise ValueError('Records do not match frozen subset selection')
    result = analyze(records)
    result['parallel_pairwise'] = {}
    for arm in ('A_prime', 'B', 'B0'):
        pair = analyze({name: records[arm] if name == 'B' else records['A'] for name in ARMS})
        result['parallel_pairwise'][arm + '_vs_A'] = {
            'losses': pair['losses_vs_R']['B'], 'first_divergence': pair['first_divergence']['B'],
            'support': pair['support'], 'censored_targets': pair['censored_targets'],
            'scope': 'same 296 positions; pairwise shared top20 KL; diagnostic, separate from R gate'}
    if result['cases'] != 35 or result['positions'] != 296:
        raise ValueError('Adoption gate requires the complete frozen 35 cases / 296 positions')
    result['evidence_sha256'] = {arm: hashlib.sha256(path.read_bytes()).hexdigest() for arm, path in paths.items()}
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
