import unittest
from weight_decode_profiles import weighted_profiles,scenario

def profile(cycle,kernel,gap):
 return {'window_ms_per_step':cycle,'steps':96,'categories':{'matrix':{'union_ms_per_step':kernel}},'named_kernel_union_ms_per_step':kernel,'uncovered_by_named_kernel_ms_per_step':gap,'fp16_module_attribution':{'groups_ms_per_step':{'qkvz':kernel}}}
class WeightTests(unittest.TestCase):
 def test_weights_fractions_before_ms_and_averages_not_sums_ranks(self):
  cases={'b1':[profile(10,8,2)]*2,'b2':[profile(100,20,80)]*2}
  r=weighted_profiles(cases,{'b1':64,'b2':36},.68,37)
  self.assertAlmostEqual(r['composition']['matrix']['decode_time_fraction'],.584)
  self.assertAlmostEqual(r['fraction_sum'],1)
  self.assertAlmostEqual(r['composition']['matrix']['equivalent_ms_per_replay_step'],.584*37)
  self.assertAlmostEqual(scenario(r['fp16_groups']['qkvz'],2)['wall_reduction_fraction'],.584*.68/2)
 def test_overlap_is_explicit_and_weights_are_normalized(self):
  p=profile(10,8,3);p['named_kernel_union_ms_per_step']=7
  r=weighted_profiles({'b1':[p,p]},{'b1':100.01},.68,37)
  self.assertAlmostEqual(r['composition']['category_overlap_correction']['decode_time_fraction'],-.1)
  self.assertAlmostEqual(r['fraction_sum'],1)
 def test_rejects_missing_rank_and_unreachable_threshold(self):
  with self.assertRaises(ValueError):weighted_profiles({'b1':[profile(1,1,0)]},{'b1':1},1,1)
  self.assertIsNone(scenario({'replay_wall_fraction':.02},2)['minimum_kernel_speedup_for_3pct_wall_ignoring_overheads'])
if __name__=='__main__':unittest.main()
