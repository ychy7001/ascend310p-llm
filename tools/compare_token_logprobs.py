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
        uncensored = [error for position, error in enumerate(case_errors)
                      if position not in case_censored]
        # Positions strictly before the first divergence: both arms choose the
        # same token there, so that token is the candidate's own top-1 and is
        # always returned. Censoring is impossible by construction, which makes
        # this the only magnitude family that needs no lower-bound caveat.
        prefix = length if divergence is None else divergence
        prefix_errors = case_errors[:prefix]
        per_case.append({
            'id': left['id'], 'positions_compared': count,
            'top1_mismatches': case_mismatch,
            'top1_mismatch_rate': case_mismatch/count,
            'top1_first_divergence': top1_diff,
            'greedy_first_divergence': divergence,
            'top1_divergent_suffix_fraction': 0 if top1_diff is None else (length-top1_diff)/length,
            'greedy_divergent_suffix_fraction': 0 if divergence is None else (length-divergence)/length,
            'max_absolute_target_logprob_difference': max(case_errors),
            'mean_conditional_topk_kl': sum(case_kls)/count,
            'max_conditional_topk_kl': max(case_kls),
            'prefix_positions': prefix,
            'prefix_mean_absolute_target_logprob_difference':
                None if not prefix_errors else sum(prefix_errors)/prefix,
            'max_uncensored_absolute_target_logprob_difference':
                max(uncensored) if uncensored else None,
            'uncensored_positions': len(uncensored),
            'minimum_shared_explicit_tokens': min(supports),
            'censored_target_positions':case_censored,
        })
    if not observed:
        raise ValueError('No observations')
    cases = len(per_case)
    means = lambda key: sum(case[key] for case in per_case)/cases
    # Two families, both measured and both reported; the rule's gating_metrics
    # names which of them decide. Measured over two independent A/A_prime rounds
    # on the same image, the first family reproduces to 1.18x and the second to
    # 1.49x, which is why the first is the gate.
    #
    # Gating family: per-case normalized, so the value does not move with the
    # number of compared positions (truncation at the first divergence makes that
    # vary between rounds), the per-case maxima are means over cases rather than
    # the single worst position of the run, and the two magnitude metrics need no
    # censoring caveat -- the prefix one is uncensorable by construction and the
    # uncensored maximum excludes the truncated positions entirely.
    #
    # Diagnostic family: the previous gate. Position-weighted means and run-wide
    # extremes are both dominated by whichever case happened to diverge that
    # round; the max logprob difference can only be a lower bound.
    covered = [case for case in per_case if case['max_uncensored_absolute_target_logprob_difference'] is not None]
    with_prefix = [case for case in per_case if case['prefix_positions']]
    metrics = {
        'top1_disagreeing_case_rate': sum(1 for case in per_case if case['top1_mismatches'])/cases,
        'mean_case_top1_mismatch_rate': means('top1_mismatch_rate'),
        'mean_case_mean_conditional_topk_kl': means('mean_conditional_topk_kl'),
        'mean_case_max_conditional_topk_kl': means('max_conditional_topk_kl'),
        'mean_prefix_absolute_target_logprob_difference':
            0.0 if not with_prefix else sum(case['prefix_mean_absolute_target_logprob_difference']
                                            for case in with_prefix)/cases,
        'mean_case_max_uncensored_target_logprob_difference':
            0.0 if not covered else sum(case['max_uncensored_absolute_target_logprob_difference']
                                        for case in covered)/cases,
        'mean_top1_divergent_suffix_fraction': means('top1_divergent_suffix_fraction'),
        'max_top1_divergent_suffix_fraction': max(case['top1_divergent_suffix_fraction'] for case in per_case),
        'mean_greedy_divergent_suffix_fraction': means('greedy_divergent_suffix_fraction'),
        'max_greedy_divergent_suffix_fraction': max(case['greedy_divergent_suffix_fraction'] for case in per_case),
        'top1_disagreement_rate': mismatches/observed,
        'mean_conditional_topk_kl': sum(kls)/observed,
        'max_conditional_topk_kl': max(kls),
        'max_absolute_target_logprob_difference': max(errors),
    }
    severity = {
        'positions_above_kl_0.5': sum(1 for value in kls if value > .5),
        'positions_above_kl_1': sum(1 for value in kls if value > 1),
        'positions_above_absolute_error_0.5': sum(1 for value in errors if value > .5),
        'positions_above_absolute_error_1': sum(1 for value in errors if value > 1),
    }
    return {'mode':reference['mode'], 'metrics':metrics, 'severity':severity,
            'top1_agreement_rate':1-mismatches/observed,
            'positions_compared':observed, 'per_case':per_case,
            'target_logprob_censoring':{'count':len(censored_positions),'positions':censored_positions,
                'max_error_metric_is_lower_bound':bool(censored_positions)},
            'target_logprob_error_coverage':{'cases_with_uncensored_positions':len(covered),
                'cases_total':cases,
                'cases_without_uncensored_positions':cases-len(covered)},
            'scope':'KL of distributions normalized on shared explicit top-k support; excludes outside support and is not full vocabulary KL; generated mode only compares identical conditioning prefixes, including the first divergent position.'}


def gating_keys(rule, candidate):
    """判定指标由规则的 gating_metrics 列出；规则未列出时判定全部已测指标。"""
    keys = list(rule.get('gating_metrics') or ())
    if not keys:
        return list(candidate['metrics'])
    unknown = sorted(set(keys)-set(candidate['metrics']))
    if unknown:
        raise ValueError('Gating metrics not measured: {}'.format(unknown))
    return keys


def censored_metrics(result):
    """本次结果里只能给出下界、因而不能据此判通过的指标。"""
    censored = set()
    if result.get('target_logprob_error_coverage', {}).get('cases_without_uncensored_positions', 0):
        censored.add('mean_case_max_uncensored_target_logprob_difference')
    if result.get('target_logprob_censoring', {}).get('count', 0):
        censored.add('max_absolute_target_logprob_difference')
    return censored


def adjudicate(noise, candidate, rule):
    if noise['mode'] != candidate['mode']:
        raise ValueError('Different modes')
    multiplier = rule['multiplier']
    if not math.isfinite(multiplier) or multiplier <= 0 or rule['absolute_floor'] != 0:
        raise ValueError('Expected positive finite noise multiplier and zero absolute floor')
    if set(noise['metrics']) != set(candidate['metrics']):
        raise ValueError('Noise and candidate measured different metrics')
    censored = censored_metrics(candidate)
    checks = {}
    for key in gating_keys(rule, candidate):
        value, floor = candidate['metrics'][key], noise['metrics'][key]
        threshold = multiplier * floor
        passed = value <= threshold
        if key in censored and passed:
            passed = None  # A bound below threshold cannot establish the true error passes.
        checks[key] = {'noise':floor, 'candidate':value, 'threshold':threshold, 'passed':passed,
                       'candidate_value_kind':'lower_bound' if key in censored else 'measured'}
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
