import json
import unittest
from unittest.mock import patch

from replay_stage_workload import stream_completion


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        for event in ({'choices': [{'text': 'answer'}]},
                      {'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}):
            yield b'data: ' + json.dumps(event).encode() + b'\n\n'
        raise AssertionError('Client waited beyond terminal usage event')


class ReplayStreamTest(unittest.TestCase):
    def test_terminal_usage_closes_stream_without_done_or_eof(self):
        with patch('replay_stage_workload.post', return_value=Response()):
            first, last, usage = stream_completion('http://test', {})
        self.assertEqual(first, last)
        self.assertEqual(usage['completion_tokens'], 3)


if __name__ == '__main__':
    unittest.main()
