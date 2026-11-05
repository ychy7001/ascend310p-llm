import json
from pathlib import Path
import tempfile
import unittest
from adjudicate_moe_load_profiles import CASES, adjudicate


class LoadAdoptionTests(unittest.TestCase):
    def write(self, path, *, gain=1.0, omit=None, wrong_tokens=False, slower_prefill=False):
        events = [dict(kind='config', repeats=3, seed=17, unit_token_ids=[17, 31],
                       cases=[[*key, value] for key, value in CASES.items()])]
        for key, output in CASES.items():
            if key == omit:
                continue
            for repeat in range(3):
                events.append(dict(kind='round', valid=True, profile=key[0],
                    prompt_tokens=key[1], concurrency=key[2], output_tokens=output,
                    repeat=repeat, prefix_delta={'prefix_cache_hits_total': 0},
                    responses=[{'usage': {'prompt_tokens': key[1],
                        'completion_tokens': output-int(wrong_tokens)}} for _ in range(key[2])],
                    request_prefill_tok_s=1000*(.9 if slower_prefill else gain),
                    aggregate_output_tok_s=100*gain,
                    median_ttft_s=1/gain, median_tpot_s=.01/gain))
        path.write_text('\n'.join(json.dumps(row) for row in events)+'\n')

    def compare(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            a, b = [Path(directory)/name for name in ('a.jsonl', 'b.jsonl')]
            self.write(a)
            self.write(b, **kwargs)
            return adjudicate(a, b)

    def test_all_four_metrics_improve_across_every_shape(self):
        result = self.compare(gain=1.1)
        self.assertTrue(result['all_measured_medians_noninferior'])
        self.assertEqual(len(result['profiles']), 14)
        self.assertFalse(result['tolerance_added'])

    def test_prefill_regression_blocks_despite_decode_gain(self):
        result = self.compare(gain=1.1, slower_prefill=True)
        self.assertFalse(result['all_measured_medians_noninferior'])
        self.assertEqual({r['metric'] for r in result['regressions']}, {'request_prefill_tok_s'})

    def test_missing_shape_rejected(self):
        with self.assertRaises(AssertionError):
            self.compare(omit=('RAG', 32768, 4))

    def test_wrong_generated_length_rejected(self):
        with self.assertRaises(AssertionError):
            self.compare(wrong_tokens=True)


if __name__ == '__main__':
    unittest.main()
