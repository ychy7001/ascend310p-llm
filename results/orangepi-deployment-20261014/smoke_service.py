"""OrangePi deployment smoke test: text, prefix reuse, and a synthetic red image."""
import base64
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path

BASE = 'http://127.0.0.1:18024'
MODEL = 'qwen36-w4a8'
OUT = Path('/data/qwen36-deploy/service-smoke.jsonl')


def post(path, body):
    started = time.perf_counter()
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    return {'elapsed_seconds': time.perf_counter() - started, 'response': result}


def metrics():
    with urllib.request.urlopen(BASE + '/metrics', timeout=10) as response:
        return [line for line in response.read().decode().splitlines()
                if 'prefix_cache' in line and not line.startswith('#')]


def red_image():
    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data +
                struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff))
    raw = b''.join(b'\x00' + b'\xff\x00\x00' * 128 for _ in range(128))
    png = (b'\x89PNG\r\n\x1a\n' +
           chunk(b'IHDR', struct.pack('!2I5B', 128, 128, 8, 2, 0, 0, 0)) +
           chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def main():
    for attempt in range(70):
        try:
            with urllib.request.urlopen(BASE + '/health', timeout=3):
                break
        except Exception:
            print(f'waiting for health: attempt {attempt + 1}', flush=True)
            time.sleep(10)
    else:
        raise RuntimeError('Service did not become healthy')
    with OUT.open('a') as out:
        def save(kind, row):
            row = {'kind': kind, **row}
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
        save('text', post('/v1/chat/completions', {
            'model': MODEL, 'messages': [{'role': 'user', 'content': 'What is 2 + 2? Reply with only the number.'}],
            'temperature': 0, 'max_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False},
        }))
        ids = post('/tokenize', {'model': MODEL, 'prompt': 'Memory calibration note. ' * 2400,
                                'add_special_tokens': False})['response']['tokens'][:2304]
        for i in range(2):
            before = metrics()
            row = post('/v1/completions', {'model': MODEL, 'prompt': ids, 'max_tokens': 32,
                                          'temperature': 0, 'cache_salt': 'orangepi-prefix-smoke-v1'})
            row.update(prefix_before=before, prefix_after=metrics())
            save('prefix_cold' if i == 0 else 'prefix_hot', row)
        save('image', post('/v1/chat/completions', {
            'model': MODEL, 'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'What is the main color of this image? Reply with one color word.'},
                {'type': 'image_url', 'image_url': {'url': red_image()}},
            ]}], 'temperature': 0, 'max_tokens': 32,
            'chat_template_kwargs': {'enable_thinking': False},
        }))


if __name__ == '__main__':
    main()
