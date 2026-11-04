#!/usr/bin/env python3
"""比较固定 token 条件下的生成/teacher-forcing 记录；仅报共同 top-k 条件分布 KL。"""
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


def conditional_topk_kl(left, right):
    """共同显式 token 支持上归一化后计算 KL；不推算舍入后的 OTHER 桶。"""
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError('No shared top-k support')
    def normalized(values):
        logs = [values[key] for key in keys]
        if any(not math.isfinite(value) for value in logs):
            raise ValueError('Nonfinite shared logprob')
        peak = max(logs)
        normalizer = peak + math.log(math.fsum(math.exp(value-peak) for value in logs))
        return [value-normalizer for value in logs]
    p, q = normalized(left), normalized(right)
    value = math.fsum(math.exp(x)*(x-y) for x,y in zip(p,q))
    return max(0.0, value), len(keys)


def compare(reference, candidate):
    if reference['mode'] != candidate['mode']:
        raise ValueError('Different observation modes')
    if reference['input_sha256'] != candidate['input_sha256']:
        raise ValueError('Different frozen inputs')
    if reference['reference_sequence_sha256'] != candidate['reference_sequence_sha256']:
        raise ValueError('Different A reference sequences')
    per_case, errors, kls, mismatches, observed = [], [], [], 0, 0
    censored_positions = []
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
        case_kls, case_errors, supports, case_censored = [], [], [], []
        for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
            target = str(left['greedy_ids'][i])
            if target not in x['logprobs']:
                raise ValueError('A reference is missing its own selected token')
            a = x['logprobs'][target]
            if target in y['logprobs']:
                b = y['logprobs'][target]
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError('Nonfinite target logprob')
                error = abs(a-b)
            else:
                # The omitted token is no more likely than the last returned top-k token.
                # Hence |lp_A-lp_candidate| >= max(0, lp_A-min(returned candidate lp)).
                if not math.isfinite(a) or not y['logprobs']:
                    raise ValueError('Cannot bound omitted target logprob')
                error = max(0.0, a-min(y['logprobs'].values()))
                item = {'id':left['id'], 'position':i, 'absolute_error_lower_bound':error}
                censored_positions.append(item); case_censored.append(i)
            kl, support = conditional_topk_kl(x['logprobs'], y['logprobs'])
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
            'mean_conditional_topk_kl': sum(case_kls)/count,
            'max_conditional_topk_kl': max(case_kls),
            'minimum_shared_explicit_tokens': min(supports),
            'censored_target_positions':case_censored,
        })
    if not observed:
        raise ValueError('No observations')
    means = lambda key: sum(case[key] for case in per_case)/len(per_case)
    metrics = {
        'top1_disagreement_rate': mismatches/observed,
        'mean_conditional_topk_kl': sum(kls)/observed,
        'max_conditional_topk_kl': max(kls),
        'max_absolute_target_logprob_difference': max(errors),
        'mean_top1_divergent_suffix_fraction': means('top1_divergent_suffix_fraction'),
        'max_top1_divergent_suffix_fraction': max(case['top1_divergent_suffix_fraction'] for case in per_case),
        'mean_greedy_divergent_suffix_fraction': means('greedy_divergent_suffix_fraction'),
        'max_greedy_divergent_suffix_fraction': max(case['greedy_divergent_suffix_fraction'] for case in per_case),
    }
    return {'mode':reference['mode'], 'metrics':metrics, 'top1_agreement_rate':1-mismatches/observed,
            'positions_compared':observed, 'per_case':per_case,
            'target_logprob_censoring':{'count':len(censored_positions),'positions':censored_positions,
                'max_error_metric_is_lower_bound':bool(censored_positions)},
            'scope':'KL of distributions normalized on shared explicit top-k support; excludes outside support and is not full vocabulary KL; generated mode only compares identical conditioning prefixes, including the first divergent position.'}


def adjudicate(noise, candidate, rule):
    if noise['mode'] != candidate['mode']:
        raise ValueError('Different modes')
    assert rule['multiplier'] == 1.5 and rule['absolute_floor'] == 0
    thresholds = {key: 1.5 * value for key, value in noise['metrics'].items()}
    checks = {}
    for key,value in candidate['metrics'].items():
        censored = (key=='max_absolute_target_logprob_difference'
                    and candidate.get('target_logprob_censoring',{}).get('count',0)>0)
        passed = value <= thresholds[key]
        if censored and passed:
            passed = None  # A bound below threshold cannot establish the true error passes.
        checks[key] = {'noise':noise['metrics'][key], 'candidate':value,
                       'threshold':thresholds[key], 'passed':passed,
                       'candidate_value_kind':'lower_bound' if censored else 'measured'}
    outcomes=[row['passed'] for row in checks.values()]
    passed=False if False in outcomes else (None if None in outcomes else True)
    return {'passed':passed, 'checks':checks}


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
