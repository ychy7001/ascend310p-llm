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


if __name__ == '__main__':
    unittest.main()
