import json
from pathlib import Path
import tempfile
import unittest

from summarize import distribution, summarize


class SummaryTest(unittest.TestCase):
    def test_quantile_and_small_sample_label(self):
        result = distribution([3, None, 1, 2])
        self.assertEqual(result['p50'], 2)
        self.assertEqual(result['p95'], 3)
        self.assertIn('screening', result['scope'])

    def test_failure_in_denominator_warmup_excluded(self):
        rows = [
            {'kind': 'warmup', 'success': True, 'elapsed_s': 100},
            {'kind': 'request', 'success': True, 'elapsed_s': 2, 'ttft_s': .5},
            {'kind': 'request', 'success': False, 'elapsed_s': 10, 'error': 'timeout'},
            {'kind': 'round', 'aggregate_output_tok_s': 4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            result = summarize(path)
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['success_rate'], .5)
        self.assertEqual(result['elapsed_s']['max'], 2)
        self.assertEqual(result['failed'], 1)


if __name__ == '__main__':
    unittest.main()
