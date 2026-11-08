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
    """共同显式 token 支持上归一化后计算 KL；不推算舍入后的 OTHER 桶。

    支持集为空返回 (None, 0)：两个高熵位置可以各自返回 20 个 token 而完全不重叠
    （实测 replay-21 第 1 位，top1 分别是 98260 与 97279），这是合法的观测结果而非
    缺陷，调用方负责记账并排除，不能中止整批比较。分歧前的位置结构上不可能为空
    ——那里两臂 top1 相同，而生成模式下 greedy 恒等于 top1，故该 token 必在双方
    返回的 top-k 内。
    """
    keys = sorted(set(left) & set(right))
    if not keys:
        return None, 0
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


def top1_margin(logprobs, top1):
    """胜出 token 与次优之间的 logprob 间距。

    单调 softmax 下等于 logit margin，故不改变判据含义。返回 None 表示该位置的
    返回集合不足以定义次优（实测每个位置恒返 20 条，此分支只是防御）。
    """
    if str(top1) not in logprobs or len(logprobs) < 2:
        return None
    others = [value for key, value in logprobs.items() if key != str(top1)]
    if not others:
        return None
    return logprobs[str(top1)] - max(others)


def compare(reference, candidate):
    if reference['mode'] != candidate['mode']:
        raise ValueError('Different observation modes')
    if reference['input_sha256'] != candidate['input_sha256']:
        raise ValueError('Different frozen inputs')
    if reference['reference_sequence_sha256'] != candidate['reference_sequence_sha256']:
        raise ValueError('Different A reference sequences')
    per_case, errors, kls, mismatches, observed = [], [], [], 0, 0
    # Prefix positions of every case pooled into one list: the gating magnitude
    # metric is a single mean over all of them, not a mean of per-case means.
    prefix_kls = []
    # How far the winning token sits ahead of the runner-up, per position. A top-1
    # flip between two tokens that are nearly tied is not the same event as a flip
    # across a wide gap, but the divergence rates count both as one.
    prefix_margin_deltas = []
    censored_positions, empty_support_positions = [], []
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
        case_margin_deltas = []
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
            # Margin is measured at the same positions as the pooled prefix
            # metrics: strictly before the first divergence, where both arms
            # condition on identical context. A position that cannot define a
            # runner-up is skipped rather than counted as a zero difference.
            margin_left = top1_margin(x['logprobs'], x['top1'])
            margin_right = top1_margin(y['logprobs'], y['top1'])
            if margin_left is not None and margin_right is not None:
                case_margin_deltas.append(abs(margin_left-margin_right))
            case_errors.append(error); supports.append(support)
            if kl is None:
                # Disjoint top-k slices: record it instead of aborting the run.
                empty_support_positions.append({'id':left['id'], 'position':i})
            else:
                case_kls.append(kl)
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
        prefix_kls.extend(case_kls[:prefix])
        prefix_margin_deltas.extend(case_margin_deltas[:prefix])
        per_case.append({
            'id': left['id'], 'positions_compared': count,
            'top1_mismatches': case_mismatch,
            'top1_mismatch_rate': case_mismatch/count,
            'top1_first_divergence': top1_diff,
            'greedy_first_divergence': divergence,
            'top1_divergent_suffix_fraction': 0 if top1_diff is None else (length-top1_diff)/length,
            'greedy_divergent_suffix_fraction': 0 if divergence is None else (length-divergence)/length,
            'max_absolute_target_logprob_difference': max(case_errors),
            'mean_conditional_topk_kl':
                sum(case_kls)/len(case_kls) if case_kls else None,
            'max_conditional_topk_kl': max(case_kls) if case_kls else None,
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
    # A case whose every compared position had an empty shared support has no KL
    # at all; it is left out of the per-case KL means rather than counted as zero.
    kl_cases = [case for case in per_case if case['mean_conditional_topk_kl'] is not None]
    # An empty shared support anywhere leaves the run-wide KL extremes undefined:
    # the true KL at those positions is unknown, so a max over the measurable ones
    # is not the run's max. The count and positions are reported instead.
    kl_incomplete = bool(empty_support_positions)
    # Two families, both measured and both reported; the rule's gating_metrics
    # names which of them decide.
    #
    # Gating family: one mean pooled over every strictly pre-divergence position,
    # plus the three per-case normalized divergence/extent rates. Measured over
    # three independent A/A_prime rounds on the same image they reproduce to
    # 1.09x / 1.17x / 1.08x / 1.07x.
    #
    # Pooling is what makes the magnitude metric stable. Averaging per case
    # instead -- the earlier design -- gives 0.0289 / 0.0243 / 0.0057 over the
    # same three rounds, a 5.1x spread: 5, 4 and 5 of the 35 cases diverge at
    # position 0, carry no prefix at all, and then still weigh 1/35 each against
    # a case with 190 compared positions.
    #
    # Diagnostic family: everything else, including the whole earlier gate.
    # Per-case magnitude means (1.5-5.1x), run-wide extremes and severity counts
    # (1.4-4.0x) are dominated by which case happened to diverge, and how far,
    # in a given round. They stay reported; they no longer decide.
    covered = [case for case in per_case if case['max_uncensored_absolute_target_logprob_difference'] is not None]
    with_prefix = [case for case in per_case if case['prefix_positions']]
    metrics = {
        'top1_disagreeing_case_rate': sum(1 for case in per_case if case['top1_mismatches'])/cases,
        'mean_case_top1_mismatch_rate': means('top1_mismatch_rate'),
        'mean_greedy_divergent_suffix_fraction': means('greedy_divergent_suffix_fraction'),
        'pooled_prefix_kl': sum(prefix_kls)/len(prefix_kls) if prefix_kls else None,
        'mean_prefix_top1_margin_delta':
            sum(prefix_margin_deltas)/len(prefix_margin_deltas) if prefix_margin_deltas else None,
        'mean_top1_divergent_suffix_fraction': means('top1_divergent_suffix_fraction'),
        'mean_case_mean_conditional_topk_kl':
            sum(case['mean_conditional_topk_kl'] for case in kl_cases)/len(kl_cases) if kl_cases else None,
        'mean_case_max_conditional_topk_kl':
            sum(case['max_conditional_topk_kl'] for case in kl_cases)/len(kl_cases) if kl_cases else None,
        'mean_prefix_absolute_target_logprob_difference':
            0.0 if not with_prefix else sum(case['prefix_mean_absolute_target_logprob_difference']
                                            for case in with_prefix)/cases,
        'mean_case_max_uncensored_target_logprob_difference':
            0.0 if not covered else sum(case['max_uncensored_absolute_target_logprob_difference']
                                        for case in covered)/cases,
        'max_top1_divergent_suffix_fraction': max(case['top1_divergent_suffix_fraction'] for case in per_case),
        'max_greedy_divergent_suffix_fraction': max(case['greedy_divergent_suffix_fraction'] for case in per_case),
        'top1_disagreement_rate': mismatches/observed,
        'mean_conditional_topk_kl':
            None if kl_incomplete else (sum(kls)/len(kls) if kls else None),
        'max_conditional_topk_kl':
            None if kl_incomplete else (max(kls) if kls else None),
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
            'prefix_positions':len(prefix_kls),
            'empty_topk_support':{'count':len(empty_support_positions),
                'positions':empty_support_positions},
            'kl_censoring':{'count':len(empty_support_positions),
                'positions':empty_support_positions,
                'metric_is_unavailable_when_nonzero':True},
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


# 取值天然不超过 1 的比例指标。噪声底 × 倍率可能越过 1，此时该指标对候选恒为
# 通过——它在判定中已失去作用，静默保留会让门禁看起来比实际更严。
UNIT_BOUNDED_METRICS = {
    'top1_disagreeing_case_rate',
    'mean_case_top1_mismatch_rate',
    'mean_greedy_divergent_suffix_fraction',
    'mean_top1_divergent_suffix_fraction',
    'max_top1_divergent_suffix_fraction',
    'max_greedy_divergent_suffix_fraction',
    'top1_disagreement_rate',
    'top1_agreement_rate',
}


def adjudicate(noise, candidate, rule):
    if noise['mode'] != candidate['mode']:
        raise ValueError('Different modes')
    multiplier = rule['multiplier']
    if not math.isfinite(multiplier) or multiplier <= 0 or rule['absolute_floor'] != 0:
        raise ValueError('Expected positive finite noise multiplier and zero absolute floor')
    if set(noise['metrics']) != set(candidate['metrics']):
        raise ValueError('Noise and candidate measured different metrics')
    explicit = set(rule.get('gating_metrics') or ())
    noise_censored, candidate_censored = censored_metrics(noise), censored_metrics(candidate)
    checks = {}
    for key in gating_keys(rule, candidate):
        value, floor = candidate['metrics'][key], noise['metrics'][key]
        if value is None or floor is None:
            # 无共同 top-k 支持的位置使该指标不可测。显式列为判定指标时必须报错
            # ——不可测的指标不能通过；未列出时只记为不可用，不参与通过与否。
            if key in explicit:
                raise ValueError('Gating metric not measurable: {}'.format(key))
            checks[key] = {'noise':floor, 'candidate':value, 'threshold':None, 'passed':None,
                           'noise_value_kind':'unavailable', 'candidate_value_kind':'unavailable'}
            continue
        threshold = multiplier * floor
        # A unit-bounded metric whose threshold has reached 1 can never fail; the
        # gate would look four-metric while actually judging fewer. Report it as
        # a configuration error instead of a silent pass.
        if key in UNIT_BOUNDED_METRICS and threshold >= 1.0:
            raise ValueError(
                'Gating metric {} has threshold {} >= 1, which its range cannot exceed; '
                'the metric cannot fail'.format(key, threshold))
        floor_is_bound, value_is_bound = key in noise_censored, key in candidate_censored
        if floor_is_bound and not value_is_bound:
            # The true noise threshold is at least this lower bound. A measured
            # candidate below it proves noninferiority; exceeding it proves
            # neither failure nor success.
            passed = True if value <= threshold else None
        elif value_is_bound:
            # A bound below threshold cannot establish the true error passes.
            passed = False if value > threshold else None
        else:
            passed = value <= threshold
        checks[key] = {'noise':floor, 'candidate':value, 'threshold':threshold, 'passed':passed,
                       'noise_value_kind':'lower_bound' if floor_is_bound else 'measured',
                       'candidate_value_kind':'lower_bound' if value_is_bound else 'measured'}
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
