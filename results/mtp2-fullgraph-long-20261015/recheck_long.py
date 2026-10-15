"""补验被过早人工中断的MTP2长请求，不重跑已完成的32K对照。"""
import gzip
import subprocess
from run import ROOT, STABLE, canary, health, quality, run, stage

NAME = 'qwen36-mtp2-matched-fullgraph-112k-20261015'
OUT = ROOT / 'mtp2-recheck'

OUT.mkdir()
try:
    run('docker', 'stop', '-t', '30', STABLE)
    stage('mtp2-recheck:startup')
    run('docker', 'start', NAME)
    health()
    quality(OUT / 'quality-before.json')
    stage('mtp2-recheck:meeting112k')
    # complete() timeout=1800s; slow GDN sampling stacks alone are not a hang.
    canary(OUT / 'meeting-114176.json', 114176)
    quality(OUT / 'quality-after.json')
finally:
    run('docker', 'stop', '-t', '30', NAME)
    logs = run('docker', 'logs', NAME, capture_output=True)
    with gzip.open(OUT / 'service.log.gz', 'wt') as log:
        log.write(logs.stdout + logs.stderr)
    stage('restoring-mtp1')
    run('docker', 'start', STABLE)
    health()
    quality(ROOT / 'restored-mtp1-quality.json')
    stage('recheck-completed-restored')
