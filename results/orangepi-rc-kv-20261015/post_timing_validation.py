"""纯计时结束后，顺序检查112K会议、真实图片与prefix复用。"""
import importlib.util
import json
from pathlib import Path
import sys
import time
import urllib.request
import base64
import struct
import zlib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from bench_server_phases import prefix_snapshot
OUT = Path(__file__).resolve().parent
BASE = 'http://orangepi:18024'


def post(body):
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def red_image(size):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    raw = (b'\x00' + b'\xff\x00\x00' * size) * size
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', size, size, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def main():
    end = time.monotonic() + 3600
    while time.monotonic() < end:
        timing = OUT / 'c16-p112000-o128.jsonl'
        if timing.exists() and any(json.loads(line).get('kind') == 'request' for line in timing.read_text().splitlines()):
            break
        time.sleep(10)
    else:
        raise RuntimeError('C16 timing not complete; no validation request sent')
    spec = importlib.util.spec_from_file_location('paired', ROOT / 'results/a2-kv-comparison-20261015/run_comparison.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT, module.BASE, module.MODEL = OUT, BASE, 'qwen36-w4a8'
    module.meeting('c16')
    base = {'model': 'qwen36-w4a8', 'temperature': 0, 'max_tokens': 32,
            'chat_template_kwargs': {'enable_thinking': False}}
    rows = []
    for size in (128, 1024):
        body = {**base, 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'What is the main color? Reply only with one English color word.'},
            {'type': 'image_url', 'image_url': {'url': red_image(size)}}]}]}
        result = post(body)
        text = result['choices'][0]['message']['content']
        rows.append({'kind': 'image', 'size': size, 'result': result, 'passed': text.strip().lower().rstrip('.') == 'red'})
        (OUT / 'c16-vision-prefix.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        assert rows[-1]['passed'], rows[-1]
    body = {**base, 'messages': [{'role': 'user', 'content': '背景资料：会议系统采集音频、汇总反馈。\n' * 350 + '\n计算19乘37，只返回整数。'}],
            'cache_salt': 'rc-c16-prefix-20261015'}
    texts = []
    for kind in ('prefix_cold', 'prefix_hot'):
        before = prefix_snapshot(BASE)
        result = post(body)
        after = prefix_snapshot(BASE)
        texts.append(result['choices'][0]['message']['content'])
        rows.append({'kind': kind, 'result': result, 'prefix_delta': {key: after[key] - before[key] for key in before}})
        (OUT / 'c16-vision-prefix.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    assert texts[0].strip() == texts[1].strip() == '703', texts
    assert rows[-1]['prefix_delta']['vllm:prefix_cache_hits_total'] > 0
    print('RC C16 112K meeting, 128/1024 images and prefix hit passed', flush=True)


if __name__ == '__main__':
    main()
