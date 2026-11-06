import copy
import math
import unittest
from compare_wy_reference import ARMS, analyze, judge, target_distance


def records():
    probs = {str(k): -1.0 - k for k in range(20)}
    result = {}
    for arm in ARMS:
        result[arm] = {'input_sha256': 'input', 'continuation_sha256': 'fresh', 'cases': [
            {'id': 'case', 'positions': [{'position': 0, 'context_sha256': 'context', 'target': 0,
                'top1': 0, 'logprobs': dict(probs), 'target_logprob': -1.0, 'target_upper_bound': None}]}]}
    return result


class ReferenceTests(unittest.TestCase):
    def test_exact_boundary_pass(self):
        self.assertTrue(judge((3, 3), (2, 2), (3, 3))['passed'])

    def test_exact_failure(self):
        self.assertFalse(judge((3.01, 3.01), (2, 2), (3, 3))['passed'])

    def test_improvement_pass(self):
        self.assertTrue(judge((1, 1), (2, 2), (2, 2))['passed'])

    def test_censored_indeterminate(self):
        self.assertIsNone(judge((1, math.inf), (2, 2), (3, 3))['passed'])

    def test_bound_proves_failure(self):
        self.assertFalse(judge((4, math.inf), (2, 2), (3, 3))['passed'])

    def test_noise_uncertain_not_exact(self):
        self.assertIsNone(judge((4, 4), (2, 2), (3, math.inf))['passed'])

    def test_missing_metric_cannot_pass(self):
        self.assertIsNone(judge(None, (0, 0), (0, 0))['passed'])

    def test_target_bounds(self):
        self.assertEqual(target_distance({'target_logprob': -1}, {'target_logprob': None, 'target_upper_bound': -8}), (7, math.inf))
        self.assertEqual(target_distance({'target_logprob': None}, {'target_logprob': None}), (0, math.inf))

    def test_identical_records(self):
        r = analyze(records())
        self.assertTrue(r['passed'])
        self.assertEqual(r['positions'], 1)
        self.assertEqual(r['support'][0]['size'], 20)

    def test_context_mismatch_rejected(self):
        r = records();r['B']['cases'][0]['positions'][0]['context_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'context'):analyze(r)

    def test_support_includes_B0(self):
        r = records();row = r['B0']['cases'][0]['positions'][0]
        row['logprobs'] = {str(k): -k for k in range(20, 40)};row['top1'] = 20
        row['target_logprob'] = None;row['target_upper_bound'] = -39
        result = analyze(r)
        self.assertEqual(result['support'][0]['size'], 0)
        self.assertIsNone(result['passed'])
        self.assertIsNone(result['checks']['mean_conditional_topk_kl']['passed'])

    def test_first_divergence_failure(self):
        r = records();row = r['B']['cases'][0]['positions'][0]
        row['logprobs']['0'] = -3;row['logprobs']['1'] = -1;row['target_logprob'] = -3;row['top1'] = 1
        result = analyze(r)
        self.assertFalse(result['passed'])
        self.assertEqual(result['first_divergence']['B']['case'], {'index': 0, 'loss': 1})

    def test_hash_mismatch(self):
        r = records();r['A_prime']['continuation_sha256'] = 'old'
        with self.assertRaisesRegex(ValueError, 'hash'):analyze(r)

    def test_diagnostic_rejected(self):
        r = records();r['R']['diagnostic_only'] = True
        with self.assertRaisesRegex(ValueError, 'Diagnostic'):analyze(r)

    def test_duplicate_candidate_rejected(self):
        r = records();r['B']['cases'].append(copy.deepcopy(r['B']['cases'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicated'):analyze(r)

    def test_bad_bound_rejected(self):
        r = records();row = r['B']['cases'][0]['positions'][0]
        row['target'] = 99
        for arm in ARMS:r[arm]['cases'][0]['positions'][0]['target'] = 99
        row['target_logprob'] = None;row['target_upper_bound'] = -100
        with self.assertRaisesRegex(ValueError, 'censoring'):analyze(r)


if __name__ == '__main__':unittest.main()
