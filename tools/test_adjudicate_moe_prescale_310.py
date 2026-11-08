#!/usr/bin/env python3
"""CPU checks for the entry screen's budget and incomplete-evidence handling."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from adjudicate_moe_prescale_310 import adjudicate


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        # Construct a pool with the archived total, independently of its
        # per-batch distribution. A uniform 2x must halve that exact pool.
        pool=.16163336554424895
        (self.root/'weighted-composition.json').write_text(json.dumps({
            'decode_wall_fraction':.7,'time_weights':{f'b{b}':.25 for b in range(1,5)}}))
        with (self.root/'per-case-composition.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['batch','rank','category','fraction_of_case_cycle'])
            writer.writeheader()
            for batch in range(1,5):
                for rank in range(2):
                    writer.writerow({'batch':batch,'rank':rank,'category':'moe_grouped_w8a8','fraction_of_case_cycle':pool/.7})
        self.rows=[]
        for batch in range(1,5):
            for bank in (100,200):
                for trial in range(3):
                    for step in range(3):
                        for k in (256,2048):
                            self.rows.append({'label':f'input-{bank}-b{batch}-trial{trial}-{step}',
                                'm':batch*8,'k':k,'graph_replay_verified':True,'parity':{'compiled_baseline':True,'candidate':True},
                                'repeat':[True]*3,'cpu_fp32':{'A':{'finite':True,'outside_tolerance':0},'B':{'finite':True,'outside_tolerance':0}},
                                'eager':{'median_us':{'A':100.,'B':50.}},'graph':{'median_us':{'A':100.,'B':50.}}})

    def test_twofold_halves_pool(self):
        result=adjudicate(self.rows,self.root)
        self.assertAlmostEqual(result['projected_replay_reduction_pct'],8.081668277212447)
        self.assertTrue(result['enter_model_gate'])
        self.assertFalse(result['performance_acceptance'])

    def test_empty_or_unverified_graph_rejected(self):
        self.rows[0]['graph_replay_verified']=False
        with self.assertRaisesRegex(AssertionError,'input-mutation'):
            adjudicate(self.rows,self.root)

    def test_equal_cost_stops(self):
        for row in self.rows:
            for mode in ('graph','eager'):row[mode]['median_us']['B']=100.
        result=adjudicate(self.rows,self.root)
        self.assertAlmostEqual(result['projected_replay_reduction_pct'],0.)
        self.assertFalse(result['enter_model_gate'])

    def test_missing_rank_bank_rejected(self):
        with self.assertRaisesRegex(AssertionError,'two independent rank banks'):
            adjudicate([r for r in self.rows if not r['label'].startswith('input-200-')],self.root)

    def test_insufficient_requests_rejected(self):
        with self.assertRaisesRegex(AssertionError,'3 requests'):
            adjudicate([r for r in self.rows if 'trial2' not in r['label']],self.root)

    def test_bitwise_difference_requires_G1_not_G0_rejection(self):
        self.rows[0]['parity']['candidate']=False
        result=adjudicate(self.rows,self.root)
        self.assertTrue(result['G0_pass'])
        self.assertTrue(result['enter_model_gate'])
        self.assertTrue(result['requires_G1'])

    def test_any_numeric_failure_stops(self):
        self.rows[0]['cpu_fp32']['B']['outside_tolerance']=1
        self.assertFalse(adjudicate(self.rows,self.root)['enter_model_gate'])


if __name__=='__main__':unittest.main()
