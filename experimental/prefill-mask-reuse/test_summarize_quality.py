import unittest

from summarize_quality import first_token_difference, max_common_prefix_logprob_delta, step_identity


class QualityEvidenceTest(unittest.TestCase):
    def test_schedule_normalizes_ids_but_keeps_execution_order(self):
        warm = 'cmpl-mq-w1-reuse-warm-followup-random'
        cold = 'cmpl-mq-w1-reuse-cold-long-other'
        step = dict(mode='reuse', counts=[[warm, 3], [cold, 1917]], order=[cold, 'inactive', warm])
        self.assertEqual(step_identity(step), (1, (('cold-long', 1917), ('warm-followup', 3))))
        self.assertIsNone(step_identity(dict(mode='reuse', counts=[['other', 3]], order=['other'])))

    def test_first_difference_uses_token_ids_and_retains_censored_alternatives(self):
        def p(token, prob, **top):
            return dict(token_id=token, selected_logprob=prob, top=top)

        a = [p(5, -0.1), p(7, -0.3, **{'7': -0.3}), p(9, -0.5)]
        b = [p(5, -0.2), p(8, -0.2, **{'7': -0.4, '8': -0.2}), p(9, -2.0)]
        difference = first_token_difference(a, b)
        self.assertEqual(difference, dict(index_zero_based=1, left_token=7, right_token=8,
                         left_logprobs={'7': -0.3, '8': None}, right_logprobs={'7': -0.4, '8': -0.2}))
        self.assertAlmostEqual(max_common_prefix_logprob_delta(a, b, difference), 0.1)
        self.assertIsNone(first_token_difference(a, a))


if __name__ == '__main__':
    unittest.main()
