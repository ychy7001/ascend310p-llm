#!/usr/bin/env python3
"""比较固定 token 条件下的生成/teacher-forcing 记录；仅报折叠 top-k KL。"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def first_divergence(left, right):
    return next((i for i in range(max(len(left), len(right)))
                 if i >= len(left) or i >= len(right) or left[i] != right[i]), None)


def folded_kl(left, right):
    """共同显式 token + 单个 OTHER 桶的 KL；同分区，是完整 KL 的下界。"""
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError('No shared top-k support')
    p = [math.exp(left[key]) for key in keys]
    q = [math.exp(right[key]) for key in keys]
    if sum(p) > 1.00001 or sum(q) > 1.00001:
        raise ValueError('Logprobs do not describe a probability distribution')
    # Only roundoff at a saturated probability can make the residual negative.
    p.append(max(0.0, 1.0 - sum(p)))
    q.append(max(0.0, 1.0 - sum(q)))
    value = sum(x * math.log(x / y) if y else math.inf
                for x, y in zip(p, q) if x)
    return max(0.0, value), len(keys)


def compare(reference, candidate):
    if reference['mode'] != candidate['mode']:
        raise ValueError('Different observation modes')
    if reference['input_sha256'] != candidate['input_sha256']:
        raise ValueError('Different frozen inputs')
    if reference['reference_sequence_sha256'] != candidate['reference_sequence_sha256']:
        raise ValueError('Different A reference sequences')
    per_case, errors, kls, mismatches, observed = [], [], [], 0, 0
    for left, right in zip(reference['cases'], candidate['cases'], strict=True):
        if left['id'] != right['id']:
            raise ValueError('Case order mismatch')
        length = len(left['greedy_ids'])
        assert length == 256 and len(right['greedy_ids']) == length
        divergence = first_divergence(left['greedy_ids'], right['greedy_ids'])
        count = length if reference['mode'] == 'teacher_forcing' else (
            length if divergence is None else divergence + 1)
        xs, ys = left['positions'][:count], right['positions'][:count]
        assert len(xs) == count and len(ys) == count
        top1_left, top1_right = [], []
        case_kls, case_errors, supports = [], [], []
        for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
            target = str(left['greedy_ids'][i])
            if target not in x['logprobs'] or target not in y['logprobs']:
                raise ValueError(f'Missing A target logprob: {left["id"]} position {i}')
            a, b = x['logprobs'][target], y['logprobs'][target]
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError('Nonfinite target logprob')
            error = abs(a - b)
            kl, support = folded_kl(x['logprobs'], y['logprobs'])
            top1_left.append(x['top1']); top1_right.append(y['top1'])
            case_errors.append(error); case_kls.append(kl); supports.append(support)
        top1_diff = first_divergence(top1_left, top1_right)
        case_mismatch = sum(a != b for a, b in zip(top1_left, top1_right))
        mismatches += case_mismatch; observed += count
        errors.extend(case_errors); kls.extend(case_kls)
        per_case.append({
            'id': left['id'], 'positions_compared': count,
            'top1_mismatches': case_mismatch,
            'top1_first_divergence': top1_diff,
            'greedy_first_divergence': divergence,
            'top1_divergent_suffix_fraction': 0 if top1_diff is None else (length-top1_diff)/length,
            'greedy_divergent_suffix_fraction': 0 if divergence is None else (length-divergence)/length,
            'max_absolute_target_logprob_difference': max(case_errors),
            'mean_folded_topk_kl': sum(case_kls)/count,
            'max_folded_topk_kl': max(case_kls),
            'minimum_shared_explicit_tokens': min(supports),
        })
    if not observed:
        raise ValueError('No observations')
    means = lambda key: sum(case[key] for case in per_case)/len(per_case)
    metrics = {
        'top1_disagreement_rate': mismatches/observed,
        'mean_folded_topk_kl': sum(kls)/observed,
        'max_folded_topk_kl': max(kls),
        'max_absolute_target_logprob_difference': max(errors),
        'mean_top1_divergent_suffix_fraction': means('top1_divergent_suffix_fraction'),
        'max_top1_divergent_suffix_fraction': max(case['top1_divergent_suffix_fraction'] for case in per_case),
        'mean_greedy_divergent_suffix_fraction': means('greedy_divergent_suffix_fraction'),
        'max_greedy_divergent_suffix_fraction': max(case['greedy_divergent_suffix_fraction'] for case in per_case),
    }
    return {'mode':reference['mode'], 'metrics':metrics, 'top1_agreement_rate':1-mismatches/observed,
            'positions_compared':observed, 'per_case':per_case,
            'scope':'Folded shared-top-k plus OTHER KL, not full vocabulary KL; generated mode only compares identical conditioning prefixes, including the first divergent position.'}


def adjudicate(noise, candidate, rule):
    if noise['mode'] != candidate['mode']:
        raise ValueError('Different modes')
    assert rule['multiplier'] == 1.5 and rule['absolute_floor'] == 0
    thresholds = {key: 1.5 * value for key, value in noise['metrics'].items()}
    checks = {key: {'noise':noise['metrics'][key], 'candidate':value,
                    'threshold':thresholds[key], 'passed':value <= thresholds[key]}
              for key,value in candidate['metrics'].items()}
    return {'passed':all(row['passed'] for row in checks.values()), 'checks':checks}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=compare(json.loads(args.reference.read_text()),json.loads(args.candidate.read_text()))
    with args.out.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


if __name__ == '__main__':
    main()
