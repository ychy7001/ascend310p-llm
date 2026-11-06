import copy
import math
import unittest

from compare_token_logprobs import adjudicate, censored_metrics, compare, first_divergence, conditional_topk_kl


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

    def test_strict_noise_floor_uses_frozen_multiplier(self):
        x=self.record();noise=compare(x,x)
        y=copy.deepcopy(x);y['cases'][0]['positions'][4]['logprobs']['1']=math.log(.79)
        paired=compare(x,y)
        noise['metrics']=dict(paired['metrics'])
        self.assertTrue(adjudicate(noise,paired,{'multiplier':1.0,'absolute_floor':0})['passed'])
        noise['metrics']['max_absolute_target_logprob_difference']/=1.25
        strict=adjudicate(noise,paired,{'multiplier':1.0,'absolute_floor':0})
        self.assertFalse(strict['passed'])
        self.assertTrue(adjudicate(noise,paired,{'multiplier':1.5,'absolute_floor':0})['passed'])
        self.assertEqual(strict['checks']['max_absolute_target_logprob_difference']['threshold'],
                         noise['metrics']['max_absolute_target_logprob_difference'])

    def test_invalid_multiplier_rejected(self):
        noise=compare(self.record(),self.record())
        for multiplier in (0,-1,math.inf,math.nan):
            with self.assertRaises(ValueError):
                adjudicate(noise,noise,{'multiplier':multiplier,'absolute_floor':0})

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
        rule={'multiplier':1.5,'absolute_floor':0,'gating_metrics':['max_absolute_target_logprob_difference']}
        noise=compare(self.record(),self.record())
        noise['metrics']['max_absolute_target_logprob_difference']=10
        y=self.record();del y['cases'][0]['positions'][0]['logprobs']['1']
        y['cases'][0]['greedy_ids'][0]=2
        y['cases'][0]['positions'][0]['top1']=2
        result=adjudicate(noise,compare(self.record(),y),rule)
        self.assertIsNone(result['checks']['max_absolute_target_logprob_difference']['passed'])
        self.assertIsNone(result['passed'])

    def test_per_case_metrics_do_not_move_with_compared_position_count(self):
        # The same absolute amount of divergence, spread over cases: the
        # position-weighted rate halves when the compared denominator doubles,
        # the per-case metrics do not.
        x=self.record('generated')
        early=copy.deepcopy(x);late=copy.deepcopy(x)
        for y,at in ((early,5),(late,250)):
            y['cases'][0]['greedy_ids'][at]=2
            y['cases'][0]['positions'][at]['top1']=2
            y['cases'][0]['positions']=y['cases'][0]['positions'][:at+1]
        e,l=compare(x,early),compare(x,late)
        self.assertAlmostEqual(e['metrics']['top1_disagreement_rate'],1/6)
        self.assertAlmostEqual(l['metrics']['top1_disagreement_rate'],1/251)
        self.assertAlmostEqual(e['metrics']['mean_case_top1_mismatch_rate'],1/6)
        self.assertAlmostEqual(l['metrics']['mean_case_top1_mismatch_rate'],1/251)
        self.assertEqual(e['metrics']['top1_disagreeing_case_rate'],1.0)
        self.assertEqual(l['metrics']['top1_disagreeing_case_rate'],1.0)

    def test_prefix_magnitude_is_uncensored_and_excludes_divergence(self):
        x=self.record('generated');y=copy.deepcopy(x)
        # Divergence at position 4: positions 0-3 are the shared prefix.
        y['cases'][0]['greedy_ids'][4]=2;y['cases'][0]['positions'][4]['top1']=2
        y['cases'][0]['positions']=y['cases'][0]['positions'][:5]
        y['cases'][0]['positions'][4]['logprobs']={'2':math.log(.9)}
        result=compare(x,y)
        self.assertEqual(result['per_case'][0]['prefix_positions'],4)
        self.assertEqual(result['per_case'][0]['censored_target_positions'],[4])
        self.assertEqual(result['target_logprob_error_coverage']['cases_without_uncensored_positions'],0)
        self.assertNotIn('mean_prefix_absolute_target_logprob_difference',censored_metrics(result))
        # Moving a prefix position does move it; moving only the divergence
        # position does not.
        moved=copy.deepcopy(y);moved['cases'][0]['positions'][2]['logprobs']['1']=math.log(.75)
        self.assertGreater(compare(x,moved)['metrics']['mean_prefix_absolute_target_logprob_difference'],
                           result['metrics']['mean_prefix_absolute_target_logprob_difference'])

    def test_gating_metrics_select_the_judged_subset(self):
        noise=compare(self.record(),self.record())
        y=copy.deepcopy(self.record());y['cases'][0]['positions'][4]['logprobs']['1']=math.log(.79)
        paired=compare(self.record(),y)
        noise['metrics']=dict(paired['metrics'])
        noise['metrics']['max_absolute_target_logprob_difference']/=1.25
        rule={'multiplier':1.0,'absolute_floor':0,
              'gating_metrics':['mean_case_max_conditional_topk_kl','mean_prefix_absolute_target_logprob_difference']}
        selected=adjudicate(noise,paired,rule)
        self.assertEqual(sorted(selected['checks']),
                         ['mean_case_max_conditional_topk_kl','mean_prefix_absolute_target_logprob_difference'])
        self.assertTrue(selected['passed'])
        with self.assertRaisesRegex(ValueError,'not measured'):
            adjudicate(noise,paired,{**rule,'gating_metrics':['not_a_metric']})

    def multi(self, cases=2):
        record={'mode':'generated','input_sha256':'frozen','reference_sequence_sha256':'A','cases':[]}
        for n in range(cases):
            position={'top1':1,'logprobs':{'1':math.log(.8),'2':math.log(.1)}}
            record['cases'].append({'id':'case%d'%n,'greedy_ids':[1]*256,
                                    'positions':[copy.deepcopy(position) for _ in range(256)]})
        return record

    def diverge(self, record, case, at):
        record['cases'][case]['greedy_ids'][at]=2
        record['cases'][case]['positions'][at]['top1']=2
        record['cases'][case]['positions']=record['cases'][case]['positions'][:at+1]
        return record

    def test_empty_shared_support_is_recorded_not_fatal(self):
        # Two high-entropy positions can each return a full top-k with no token in
        # common. That is an observation, not a defect: the run must finish and say so.
        x=self.multi(1);y=self.diverge(copy.deepcopy(x),0,1)
        y['cases'][0]['positions'][1]['logprobs']={'9':math.log(.7)}
        self.assertEqual(conditional_topk_kl(x['cases'][0]['positions'][1]['logprobs'],
                                             y['cases'][0]['positions'][1]['logprobs']),(None,0))
        result=compare(x,y)
        self.assertEqual(result['empty_topk_support']['count'],1)
        self.assertEqual(result['empty_topk_support']['positions'],[{'id':'case0','position':1}])
        self.assertEqual(result['prefix_positions'],1)
        self.assertIsNotNone(result['metrics']['pooled_prefix_kl'])

    def test_pooled_prefix_kl_weights_by_position_not_by_case(self):
        # One long-prefix case and one short-prefix case, with a disturbance inside
        # the long prefix only. Pooled is the mean over all 8 prefix positions, so
        # that one position contributes 1/8 -- not 1/2, which is what averaging the
        # two per-case means would give it.
        x=self.multi(2)
        y=self.diverge(self.diverge(copy.deepcopy(x),0,6),1,2)
        y['cases'][0]['positions'][3]['logprobs']['1']=math.log(.5)
        result=compare(x,y)
        self.assertEqual(result['prefix_positions'],8)
        self.assertEqual([case['prefix_positions'] for case in result['per_case']],[6,2])
        disturbed=conditional_topk_kl(x['cases'][0]['positions'][3]['logprobs'],
                                      y['cases'][0]['positions'][3]['logprobs'])[0]
        self.assertGreater(disturbed,0)
        self.assertAlmostEqual(result['metrics']['pooled_prefix_kl'],disturbed/8,places=12)
        # Averaging the two per-case prefix means instead gives that same disturbed
        # position weight 1/6 within its case and then 1/2 across cases, i.e.
        # disturbed/12 -- a value the round it was measured in, not the run, decides.
        per_case=(disturbed/6 + 0)/2
        self.assertNotAlmostEqual(result['metrics']['pooled_prefix_kl'],per_case,places=12)

    def test_prefix_margin_delta_measures_runner_up_gap(self):
        # Margin is the distance from the winner to the runner-up, so the same
        # winner with a reshaped tail still moves it.
        x=self.record('generated')
        self.assertAlmostEqual(compare(x,x)['metrics']['mean_prefix_top1_margin_delta'],0)
        y=copy.deepcopy(x)
        for position in y['cases'][0]['positions']:
            position['logprobs']={'1':math.log(.7),'2':math.log(.2)}
        delta=compare(x,y)['metrics']['mean_prefix_top1_margin_delta']
        self.assertAlmostEqual(delta,abs((math.log(.8)-math.log(.1))-(math.log(.7)-math.log(.2))),places=12)
        # It is confined to the shared prefix: a divergence position does not count.
        self.assertAlmostEqual(
            compare(x,self.diverge(copy.deepcopy(x),0,3))['metrics']['mean_prefix_top1_margin_delta'],0)

    def test_unit_bounded_gate_above_its_range_is_rejected(self):
        # A rate cannot exceed 1, so a threshold of 1.18 can never fail. Silently
        # judging it would make the gate look stricter than it is.
        noise=compare(self.multi(1),self.multi(1))
        noise['metrics']['mean_case_top1_mismatch_rate']=0.95
        rule={'multiplier':1.40,'absolute_floor':0,'gating_metrics':['mean_case_top1_mismatch_rate']}
        with self.assertRaisesRegex(ValueError,'cannot fail'):
            adjudicate(noise,noise,dict(rule))
        # The same metric stays judgeable while its threshold remains in range.
        noise['metrics']['mean_case_top1_mismatch_rate']=0.5
        self.assertTrue(adjudicate(noise,noise,dict(rule))['passed'])

    def test_unmeasurable_gating_metric_cannot_pass(self):
        # Every case diverges at position 0, so no prefix position exists and the
        # pooled magnitude is undefined. Judging on it must fail loudly, not pass.
        x=self.multi(1)
        result=compare(x,self.diverge(copy.deepcopy(x),0,0))
        self.assertEqual(result['prefix_positions'],0)
        self.assertIsNone(result['metrics']['pooled_prefix_kl'])
        rule={'multiplier':1.25,'absolute_floor':0,'gating_metrics':['pooled_prefix_kl']}
        with self.assertRaisesRegex(ValueError,'not measurable'):
            adjudicate(result,result,rule)


if __name__=='__main__':unittest.main()
