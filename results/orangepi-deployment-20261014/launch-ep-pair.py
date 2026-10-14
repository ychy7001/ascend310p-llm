"""复用a2已保存配置启动等参候选；原服务保留并可用原CreateCommand恢复。"""
import json
import subprocess
from pathlib import Path

work = Path('/home/ychy/ascend-lab-runtime/rc-ep-pair-20261014')
info = json.loads((work/'base-inspect.json').read_text())[0]
cmd = info['Config']['CreateCommand'].copy()
cmd[cmd.index('--name')+1] = 'qwen36-ep-rc-pair'
overrides = {'PORT':'18030', 'NUM_SPECULATIVE_TOKENS':'1', 'MAX_BATCHED_TOKENS':'2048',
             'ENABLE_PREFIX_CACHING':'1', 'OMP_NUM_THREADS':'4', 'ENABLE_PROFILER':'0'}
for i, value in enumerate(cmd):
    if i and cmd[i-1] == '-e':
        key = value.split('=',1)[0]
        if key in overrides:
            cmd[i] = key+'='+overrides[key]
    if '/home/ychy/services/qwen36/repo/' in value:
        cmd[i] = value.replace('/home/ychy/services/qwen36/repo/',str(work/'code')+'/')
script = cmd[-1]
script = script.replace('exec vllm serve /model', 'exec vllm serve /model --no-async-scheduling --kv-cache-memory-bytes 6442450944 --mm-processor-cache-gb 0.5 --mm-processor-kwargs \'{"max_pixels":1048576}\' --limit-mm-per-prompt \'{"image":1,"video":0}\'')
cmd[-1] = script
(work/'candidate-command.json').write_text(json.dumps(cmd,indent=2))
with (work/'candidate-launch.log').open('a') as log:
    subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Started isolated EP candidate on 18030; original command retained in base-inspect.json')
