"""Exercise fixed arrivals against a streaming HTTP stub with a slow first call."""
import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest


class ReplayTest(unittest.TestCase):
    def test_later_arrival_does_not_wait_for_first_completion(self):
        arrivals = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'vllm:prefix_cache_hits_total{model="stub"} 0\nvllm:prefix_cache_queries_total{model="stub"} 0\n')
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.end_headers()
                if self.path == '/tokenize':
                    self.wfile.write(b'{"tokens":[11,12,13]}')
                    return
                arrivals.append(time.monotonic())
                # The second arrival is due before the first request completes.
                if body['max_tokens'] == 2:
                    time.sleep(.5)
                self.wfile.write(('data: '+json.dumps({'id':'stub-'+str(body['max_tokens']),
                    'choices':[{'text':'ok'}], 'usage':{'prompt_tokens':len(body['prompt']),
                    'completion_tokens':body['max_tokens']}})+'\n\n').encode())
        server = http.server.ThreadingHTTPServer(('127.0.0.1',0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = {'requests':[{'label':'replay-000','arrival_seconds':0,'input_tokens':7,'output_tokens':2},
                                    {'label':'replay-001','arrival_seconds':.1,'input_tokens':9,'output_tokens':3}]}
                (root/'plan.json').write_text(json.dumps(plan))
                command = [sys.executable,str(Path(__file__).with_name('replay_frozen_arrivals.py')),
                           '--base-url',f'http://127.0.0.1:{server.server_port}', '--model','stub',
                           '--plan',str(root/'plan.json'),'--out',str(root/'events.jsonl')]
                result = subprocess.run(command,capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,0,result.stderr)
                events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
                self.assertEqual(len(arrivals),2)
                self.assertLess(arrivals[1]-arrivals[0],.4)
                self.assertGreater(arrivals[1]-arrivals[0],.05)
                self.assertEqual(events[-1]['output_tokens'],5)
                self.assertEqual(events[-1]['input_tokens'],16)
                self.assertEqual(events[-1]['requests'],2)
                self.assertEqual(sum(e['kind']=='request_identity' for e in events),2)
                self.assertEqual(sum(e['kind']=='request_end' for e in events),2)
                # Refuse accidental reuse of an existing evidence file.
                again = subprocess.run(command,capture_output=True,text=True,timeout=10)
                self.assertNotEqual(again.returncode,0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
