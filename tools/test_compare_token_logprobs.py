import copy
import math
import unittest

from compare_token_logprobs import adjudicate, compare, first_divergence, conditional_topk_kl


class NumericalComparisonTest(unittest.TestCase):
    def record(self, mode='teacher_forcing'):
        position={'top1':1,'logprobs':{'1':math.log(.8),'2':math.log(.1)}}
        return {'mode':mode,'input_sha256':'frozen','reference_sequence_sha256':'A',
                'cases':[{'id':'one','greedy_ids':[1]*256,
                          'positions':[copy.deepcopy(position) for _ in range(256)]}]}

    def test_length_divergence(self):
        self.assertEqual(first_divergence([1],[1,2]),1)
        self.assertIsNone(first_divergence([1],[1]))

    def test_shared_support_conditional_kl(self):
        kl,n=conditional_topk_kl({'a':math.log(.7),'b':math.log(.2),'c':math.log(.1)},
                                {'a':math.log(.5),'b':math.log(.4),'d':math.log(.1)})
        self.assertEqual(n,2)
        self.assertAlmostEqual(kl,7/9*math.log(7/5)+2/9*math.log(2/4))

    def test_rounded_explicit_mass_above_one(self):
        left={'a':math.log(.99999998),'b':math.log(1e-8)}
        right={'a':0.0,'b':math.log(3e-8)}
        kl,_=conditional_topk_kl(left,right)
        self.assertTrue(math.isfinite(kl))
        self.assertGreaterEqual(kl,0)
        self.assertEqual(conditional_topk_kl(right,right)[0],0)

    def test_identity_and_zero_noise_gate(self):
        x=self.record();noise=compare(x,x)
        self.assertTrue(all(value==0 for value in noise['metrics'].values()))
        y=copy.deepcopy(x);y['cases'][0]['positions'][4]['logprobs']['1']=math.log(.79)
        result=adjudicate(noise,compare(x,y),{'multiplier':1.5,'absolute_floor':0})
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['max_absolute_target_logprob_difference']['passed'])

    def test_generated_only_compares_common_context(self):
        x=self.record('generated');y=copy.deepcopy(x)
        y['cases'][0]['greedy_ids'][2]=2
        y['cases'][0]['positions'][2]['top1']=2
        y['cases'][0]['positions']=y['cases'][0]['positions'][:3]
        result=compare(x,y)
        self.assertEqual(result['positions_compared'],3)
        self.assertEqual(result['per_case'][0]['greedy_first_divergence'],2)
        self.assertAlmostEqual(result['metrics']['top1_disagreement_rate'],1/3)

    def test_missing_target_lower_bound_and_different_reference(self):
        x=self.record();y=copy.deepcopy(x)
        del y['cases'][0]['positions'][1]['logprobs']['1']
        y['cases'][0]['greedy_ids'][1]=2
        y['cases'][0]['positions'][1]['top1']=2
        result=compare(x,y)
        self.assertEqual(result['target_logprob_censoring']['count'],1)
        self.assertAlmostEqual(result['metrics']['max_absolute_target_logprob_difference'],math.log(8))
        gate=adjudicate(compare(x,x),result,{'multiplier':1.5,'absolute_floor':0})
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['checks']['max_absolute_target_logprob_difference']['candidate_value_kind'],'lower_bound')
        y=copy.deepcopy(x);y['reference_sequence_sha256']='different'
        with self.assertRaisesRegex(ValueError,'reference sequences'):
            compare(x,y)
    def test_censored_error_cannot_pass_from_a_small_bound(self):
        noise=compare(self.record(),self.record())
        noise['metrics']['max_absolute_target_logprob_difference']=10
        for key in noise['metrics']:
            if 'divergent_suffix_fraction' in key or key=='top1_disagreement_rate':noise['metrics'][key]=1
        y=self.record();del y['cases'][0]['positions'][0]['logprobs']['1']
        y['cases'][0]['greedy_ids'][0]=2
        y['cases'][0]['positions'][0]['top1']=2
        result=adjudicate(noise,compare(self.record(),y),{'multiplier':1.5,'absolute_floor':0})
        self.assertIsNone(result['checks']['max_absolute_target_logprob_difference']['passed'])
        self.assertIsNone(result['passed'])


if __name__=='__main__':unittest.main()
