"""Model adoption checks require valid observations and sufficient paired gains."""
import unittest
from unittest.mock import patch
import adjudicate_moe_model as model


def sample(wall=100,ttft=1):
    return dict(wall_s=wall,requests=[dict(label=f'replay-{i:03d}',prompt_tokens=4096,
        concurrency=1,ttft_s=ttft,mean_tpot_s=.03) for i in range(11)],
        phase_seconds={'decode':wall*.8},plan_sha256='frozen',seed=17,
        unit_token_ids=[1,2],max_arrival_lateness_s=.001)


class AdoptionTest(unittest.TestCase):
    def judge(self,gains,regress=False):
        def summarize(root,label):
            pair=int(label[1])-1
            return sample(100 if label.endswith('A') else 100*(1-gains[pair]/100),
                          1.04 if regress and label.endswith('B') else 1)
        with patch.object(model,'summarize',side_effect=summarize):
            return model.adjudicate(None)

    def test_five_good_pairs_pass(self):
        result=self.judge([4,4,4,4,4])
        self.assertTrue(result['replay_performance_pass'])
        self.assertEqual(result['faster_pairs'],5)
        self.assertAlmostEqual(result['median_wall_reduction_pct'],4)

    def test_below_threshold_fails(self):
        self.assertFalse(self.judge([2,2,2,2,2])['replay_performance_pass'])

    def test_insufficient_faster_pairs_fails(self):
        self.assertFalse(self.judge([5,5,5,-1,-1])['replay_performance_pass'])

    def test_ttft_regression_blocks_wall_gain(self):
        result=self.judge([5,5,5,5,5],regress=True)
        self.assertFalse(result['replay_performance_pass'])
        self.assertTrue(result['stratum_regressions_over_3pct'])

    def test_invalid_timing_cannot_pass(self):
        for value in (float('nan'), float('inf'), -float('inf'), 0, -1, True):
            for metric in ('wall_s', 'ttft_s', 'mean_tpot_s'):
                with self.subTest(value=value, metric=metric):
                    row = sample()
                    if metric == 'wall_s':row[metric] = value
                    else:row['requests'][0][metric] = value
                    with patch.object(model, 'summarize', return_value=row):
                        with self.assertRaisesRegex(ValueError, 'finite positive'):
                            model.adjudicate(None)


if __name__=='__main__':
    unittest.main()
