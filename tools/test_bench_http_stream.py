# SPDX-License-Identifier: Apache-2.0
"""验证流式基准不会将断流或不完整输出记为成功。"""
import io
import json
import unittest
from unittest.mock import patch

from bench_http_stream import request


class StreamCompletionTest(unittest.TestCase):
    def run_stream(self, events):
        payload = b''.join(b'data: ' + json.dumps(event).encode() + b'\n\n'
                           for event in events)
        with patch('bench_http_stream.post', return_value=io.BytesIO(payload)):
            return request('http://test', 'model', [1, 2], 3)

    def test_final_usage_without_done(self):
        result = self.run_stream([
            {'choices': [{'text': 'answer'}]},
            {'usage': {'prompt_tokens': 2, 'completion_tokens': 3}},
        ])
        self.assertEqual(result['text'], 'answer')
        self.assertIsNotNone(result['ttft_s'])

    def test_disconnect_without_final_usage(self):
        with self.assertRaisesRegex(RuntimeError, 'Incomplete benchmark response'):
            self.run_stream([{'choices': [{'text': 'partial'}]}])

    def test_partial_usage(self):
        with self.assertRaisesRegex(RuntimeError, 'Incomplete benchmark response'):
            self.run_stream([{'usage': {'prompt_tokens': 2, 'completion_tokens': 1}}])


class MeasuredRequestTest(unittest.TestCase):
    def test_failure_retained(self):
        from bench_http_stream import measured_request
        with patch('bench_http_stream.request', side_effect=RuntimeError('connection lost')):
            result = measured_request('http://test', 'model', [1, 2], 3)
        self.assertFalse(result['success'])
        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertGreaterEqual(result['elapsed_s'], 0)

    def test_seed_and_salt_forwarded(self):
        from bench_http_stream import request
        payload = b'data: {"usage":{"prompt_tokens":2,"completion_tokens":3}}\n\n'
        with patch('bench_http_stream.post', return_value=io.BytesIO(payload)) as post:
            request('http://test', 'model', [1, 2], 3, seed=310, cache_salt='run-request')
        self.assertEqual(post.call_args.args[2]['seed'], 310)
        self.assertEqual(post.call_args.args[2]['cache_salt'], 'run-request')


if __name__ == '__main__':
    unittest.main()
