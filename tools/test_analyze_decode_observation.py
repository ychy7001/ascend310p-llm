import unittest
from analyze_decode_observation import join_steps

class JoinTests(unittest.TestCase):
    def test_async_overlap_clipped_and_phase_not_guessed_from_batch(self):
        events=[]
        for step,phase,t in [(1,'prefill',10),(2,'decode',10.1),(3,'decode',20)]:
            req={'id':'one','scheduled':1,'phase':phase,'computed':step,'prompt':2}
            events.extend([
                {'kind':'schedule','mono':t-.1,'start':t-.11,'end':t-.1,'step':step,'requests':[req]},
                {'kind':'execute_start','mono':t,'pid':8,'rank':0,'seq':step,'requests':{'one':1}},
                {'kind':'forward_start','mono':t+.01,'pid':8,'seq':step,'capturing':False,'padded_tokens':1,'graph_hit':phase=='decode','mode':'FULL'},
                {'kind':'execute_return','mono':t+.05,'pid':8,'seq':step,'end':t+.05},
                {'kind':'completion','mono':t+.2,'start':t+.19,'end':t+.2,'step':step,'requests':[]}])
        steps,validation=join_steps(events)
        self.assertAlmostEqual(steps[0]['duration_s'],.2)
        self.assertAlmostEqual(steps[1]['duration_s'],.1)
        self.assertAlmostEqual(steps[2]['duration_s'],.2)
        self.assertEqual([s['phase'] for s in steps],['prefill','decode','decode'])
        self.assertEqual(validation['unmatched_worker_sequences'],[])
        self.assertLess(sum(s['duration_s'] for s in steps),1)

if __name__=='__main__':unittest.main()
