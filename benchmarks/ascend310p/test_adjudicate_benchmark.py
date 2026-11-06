import copy
import json
from pathlib import Path
import tempfile
import unittest

from adjudicate_benchmark import adjudicate, read_cell


class BenchmarkAdjudicationTest(unittest.TestCase):
    def cells(self, count=3):
        return [
            {'prompt_tokens': i + 2, 'concurrency': 1,
             **{arm: {'ttft_s': float(i + 1), 'mean_tpot_s': .01 * (i + 1),
                      'aggregate_output_tok_s': 100.0 * (i + 1),
                      'prompt_ids': list(range(i + 2))}
                for arm in ('A', 'B', 'A_prime')}}
            for i in range(count)
        ]

    def rule(self, count=3):
        return {'latency_delta': .02, 'throughput_delta': .01,
                't_critical': {'degrees_of_freedom': count - 1, 'value': 2.92}}

    def test_identical_arms(self):
        result = adjudicate(self.cells(), self.rule())
        self.assertTrue(result['passed'])
        for row in result['metrics'].values():
            self.assertEqual(row['pooled_ratio'], 1)
            self.assertEqual(row['bound'], 1)
            self.assertTrue(row['passed'])

    def test_drift_cancellation(self):
        cells = self.cells()
        for cell in cells:
            for metric in ('ttft_s', 'mean_tpot_s', 'aggregate_output_tok_s'):
                cell['A_prime'][metric] = cell['A'][metric] * 1.05
                cell['B'][metric] = cell['A'][metric] * 1.025
        self.assertTrue(adjudicate(cells, self.rule())['passed'])

    def test_real_latency_regression(self):
        cells = self.cells()
        for cell in cells:
            cell['B']['ttft_s'] *= 1.03
        result = adjudicate(cells, self.rule())
        self.assertFalse(result['passed'])
        self.assertEqual({key: row['passed'] for key, row in result['metrics'].items()},
                         {'ttft_s': False, 'mean_tpot_s': True, 'aggregate_output_tok_s': True})

    def test_latency_below_detection_limit(self):
        cells = self.cells()
        for cell in cells:
            cell['B']['ttft_s'] *= 1.01
        self.assertTrue(adjudicate(cells, self.rule())['passed'])

    def test_throughput_direction(self):
        for factor, passed in ((.98, False), (.995, True)):
            with self.subTest(factor=factor):
                cells = self.cells()
                for cell in cells:
                    cell['B']['aggregate_output_tok_s'] *= factor
                result = adjudicate(cells, self.rule())
                self.assertEqual(result['passed'], passed)
                self.assertEqual(result['metrics']['aggregate_output_tok_s']['passed'], passed)
                self.assertTrue(result['metrics']['ttft_s']['passed'])
                self.assertTrue(result['metrics']['mean_tpot_s']['passed'])

    def test_wrong_degrees_of_freedom(self):
        rule = self.rule(6)
        rule['t_critical']['degrees_of_freedom'] = 7
        with self.assertRaisesRegex(ValueError, 'observed 7'):
            adjudicate(self.cells(6), rule)

    def test_extra_rule_key(self):
        rule = self.rule()
        rule['extra'] = 1
        with self.assertRaisesRegex(ValueError, 'extra'):
            adjudicate(self.cells(), rule)

    def test_read_cell_medians_and_incomplete_evidence(self):
        rows = [{'kind': 'config', 'prompt_tokens': 2, 'concurrency': 2,
                 'prompt_ids': [10, 20]}]
        rows += [{'kind': 'request', 'success': True, 'ttft_s': value,
                  'mean_tpot_s': value / 10,
                  'usage': {'prompt_tokens': 2, 'completion_tokens': 256}}
                 for value in (1, 3, 5, 7, 9, 11)]
        rows += [{'kind': 'round', 'aggregate_output_tok_s': value}
                 for value in (10, 30, 20)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cell.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(read_cell(path, 2, 2),
                             {'ttft_s': 6.0, 'mean_tpot_s': .6,
                              'aggregate_output_tok_s': 20.0, 'prompt_ids': [10, 20]})
            for change in ('failure', 'output_tokens', 'round', 'prompt_ids'):
                with self.subTest(change=change):
                    invalid = copy.deepcopy(rows)
                    if change == 'failure':
                        invalid[1]['success'] = False
                    elif change == 'output_tokens':
                        invalid[1]['usage']['completion_tokens'] = 255
                    elif change == 'round':
                        invalid.pop()
                    else:
                        invalid[0]['prompt_ids'] = [10]
                    path.write_text(''.join(json.dumps(row) + '\n' for row in invalid))
                    with self.assertRaisesRegex(ValueError, 'observed'):
                        read_cell(path, 2, 2)

    def test_different_prompt_ids(self):
        cells = self.cells()
        cells[0]['B']['prompt_ids'] = [20, 10]
        with self.assertRaisesRegex(ValueError, 'different prompt_ids'):
            adjudicate(cells, self.rule())

    def test_duplicate_cell(self):
        cells = self.cells()
        cells[1] = copy.deepcopy(cells[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate cell'):
            adjudicate(cells, self.rule())


if __name__ == '__main__':
    unittest.main()
