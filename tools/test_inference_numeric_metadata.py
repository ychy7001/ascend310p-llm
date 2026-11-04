import json
import unittest
from inference_numeric_metadata import parse_line,sampled_histogram,quantiles

class NumericMetadataTests(unittest.TestCase):
    def test_completion_privacy(self):
        line='2026-11-04T10:00:00Z INFO Request secret-id finished: finish_reason=stop prompt_tokens=4096 generated_tokens=200 PRIVATE_BODY'
        result=parse_line(line)
        self.assertEqual(result[0]['prompt_tokens'],4096)
        self.assertNotIn('secret',json.dumps(result))
        self.assertNotIn('PRIVATE',json.dumps(result))
        self.assertEqual(parse_line('prompt text prompt_tokens=5 generated_tokens=6'),[])

    def test_production_completion_and_duration(self):
        row=parse_line('2026-11-04T10:00:00Z INFO Request finished: req secret finish_reason=stop prompt_tokens=12000 generated_tokens=700 elapsed_s=31.42')[0]
        self.assertEqual(row['elapsed_s'],31.42)
        self.assertEqual(row['generated_tokens'],700)

    def test_time_histogram_does_not_fill_missing_logs(self):
        rows=[{'time':f'2026-11-04T10:{m:02d}:{s:02d}Z','running':n} for m,s,n in [(0,0,1),(0,10,2),(0,20,0),(2,0,3)]]
        hist=sampled_histogram(rows)
        self.assertEqual(hist['covered_seconds'],20)
        self.assertEqual(hist['unobserved_seconds'],100)
        self.assertEqual(hist['seconds'],{1:10,2:10})

    def test_known_quantile(self):
        self.assertEqual(quantiles([1,2,3,4])['p50'],2.5)

if __name__=='__main__':unittest.main()
