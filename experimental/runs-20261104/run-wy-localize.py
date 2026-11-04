#!/usr/bin/env python3
"""固定生产参数的一条观测请求；finally 恢复原服务，禁止启动图像服务。"""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import time

spec=importlib.util.spec_from_file_location('model_gate',Path(__file__).with_name('run-wy-model-gate.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Localize(m.Gate):
    output_name='wy921-localize-v1'

    def model_sequence(self, unused=None):
        self.launch('capture','A')
        directory=self.out/'capture'
        for name in self.case_ids:
            case=next(c for c in self.inputs['cases'] if c['id']==name)
            marker=directory/'capture-request.json'
            m.write(marker,{'id':name,'tokens':case['tokens'],'input_sha256':self.rule['input_sha256']})
            payload={'model':m.MODEL,'prompt':case['token_ids'],'temperature':0,'seed':0,
                     'max_tokens':1,'ignore_eos':True,'return_token_ids':True,
                     'cache_salt':'wy921-localize-v1-'+name}
            m.write(self.out/(name+'-request.json'),payload)
            response=m.post(m.BASE,'/v1/completions',payload)
            m.write(self.out/(name+'-response.json'),response)
            marker.unlink()
        self.stop('capture')

    def execute(self):
        try:
            assert self.health('http://127.0.0.1:18081')
            inactive=subprocess.run(['systemctl','--user','is-active','qwen36-image.service'],capture_output=True,text=True)
            assert inactive.stdout.strip()=='inactive',inactive.stdout
            assert not subprocess.check_output(['podman','ps','-q'],text=True).strip()
            m.write(self.out/'tool-freeze.json',{p.name:m.sha(p) for p in self.tools.iterdir() if p.is_file()})
            self.paused=True
            subprocess.run(['sudo','-n','systemctl','stop',m.SERVICE],check=True)
            idle=subprocess.check_output(['npu-smi','info'],text=True)
            (self.out/'idle-before.txt').write_text(idle)
            assert 'No running processes found' in idle
            self.model_sequence()
        except Exception as error:
            self.error=repr(error);self.state('failed',error=self.error)
        finally:
            self.finish()
            state=subprocess.run(['systemctl','--user','is-active','qwen36-image.service'],capture_output=True,text=True)
            m.write(self.out/'image-service-final.json',{'state':state.stdout.strip(),'started_by_experiment':False})
            assert state.stdout.strip()=='inactive'
        if self.error:raise RuntimeError(self.error)

p=argparse.ArgumentParser(description=__doc__)
for name in ('root','tools','fixtures','dataset'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--case-ids',nargs='+',required=True)
a=p.parse_args()
signal.signal(signal.SIGTERM,m.interrupted);signal.signal(signal.SIGINT,m.interrupted)
x=Localize(a);x.case_ids=a.case_ids;x.execute()
