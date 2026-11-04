import io
import json
import unittest
from unittest.mock import patch

from bench_prefix_reuse import request, PHASE_METRICS


class PrefixStreamTest(unittest.TestCase):
    def test_usage_ends_stream_and_enforces_fixed_output(self):
        events = [{'choices': [{'text': 'answer'}]},
                  {'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}]
        data = b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events)
        before = {k: 0 for k in PHASE_METRICS}
        after = dict(zip(PHASE_METRICS, [1, 1, 1]))
        with patch('bench_prefix_reuse.stream_post', return_value=io.BytesIO(data)) as post, \
                patch('bench_prefix_reuse.metrics', side_effect=[(before, []), (after, [])]):
            result = request('http://test', 'model', [1, 2], 3, 'run', ignore_eos=True)
        self.assertTrue(result['valid'])
        self.assertEqual(post.call_args.args[2]['cache_salt'], 'run')
        self.assertTrue(post.call_args.args[2]['ignore_eos'])

    def test_incomplete_stream_is_failure(self):
        before = {k: 0 for k in PHASE_METRICS}
        with patch('bench_prefix_reuse.stream_post', return_value=io.BytesIO(b'')), \
                patch('bench_prefix_reuse.metrics', return_value=(before, [])):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete prefix'):
                request('http://test', 'model', [1, 2], 3)


if __name__ == '__main__':
    unittest.main()
