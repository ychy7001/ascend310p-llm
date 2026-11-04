#!/usr/bin/env python3
"""已知 prompt-logprob OOM 后的授权替代观测；复用 P 任务结果，新跑 A/A′/B。"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import signal
import subprocess
import time

spec=importlib.util.spec_from_file_location('wy_model_gate',Path(__file__).with_name('run-wy-model-gate.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class GeneratedGate(module.Gate):
    output_name='wy921-model-gate-generated-v1'

    def __init__(self,args):
        root=args.root.resolve();previous=root/'wy921-model-gate-v2'
        previous_status=module.read(previous/'status.json')
        assert previous_status['state']=='failed'
        restored=module.read(previous/'service-restoration.json')
        assert restored['health_200'] and restored['active_state']=='active'
        log=previous/'emergency-server.log'
        assert 'Tried to allocate 3.55 GiB' in log.read_text()
        assert 'prompt_logprobs=20' in log.read_text()
        assert not (previous/'A-G1.json').exists() and not (previous/'B').exists()
        out=root/self.output_name;out.mkdir(exist_ok=False)
        shutil.copy2(previous/'inputs.json',out/'inputs.json')
        rule=module.read(previous/'threshold-rule.json')
        rule.update(observation_mode='generated',
                    previous_threshold_rule_sha256=module.sha(previous/'threshold-rule.json'),
                    prompt_logprobs_probe='Under unchanged production memory settings the longest-input prompt-logprob probe OOMed and terminated A. Engine restored; no teacher-forcing retry. Use user-authorized generated logprobs on identical conditioning prefixes.',
                    ordering='Reuse completed P G2; fresh A G1+G2; fresh A_prime G1; freeze concrete thresholds; B G1+G2; only if both gates pass P-A-B-A performance')
        module.write(out/'threshold-rule.json',rule)
        module.write(out/'observation-mode.json',{
            'mode':'generated','reason':'Prompt-logprob availability probe OOM under fixed production parameters; API presence is not denied.',
            'probe_tokens':33024,'prompt_logprobs':20,'allocation_requested_GiB':3.55,'free_GiB':1.79,
            'probe_error_sha256':module.sha(previous/'prompt-logprobs-probe-error.json'),
            'probe_server_log_sha256':module.sha(log),'frozen_before_B':True})
        shutil.copytree(previous/'P',out/'P',ignore=lambda path,names:set(names)&{'cache'})
        for path in previous.glob('P-*'):
            if path.is_file():shutil.copy2(path,out/path.name)
        for name in ['status.json','service-restoration.json','prompt-logprobs-probe-request.json',
                     'prompt-logprobs-probe-error.json','emergency-server.log']:
            shutil.copy2(previous/name,out/('probe-attempt-'+name))
        super().__init__(args)
        self.mode='generated'
        self.state('generated_mode_and_rule_frozen',threshold_rule_sha256=module.sha(out/'threshold-rule.json'))

    def execute(self):
        try:
            assert self.health('http://127.0.0.1:18081')
            assert module.sha(self.dataset)=='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
            module.write(self.out/'tool-freeze.json',{p.name:module.sha(p) for p in self.tools.iterdir() if p.is_file()})
            self.paused=True
            subprocess.run(['sudo','-n','systemctl','stop',module.SERVICE],check=True)
            idle=subprocess.check_output(['npu-smi','info'],text=True)
            (self.out/'idle-before.txt').write_text(idle)
            assert 'No running processes found' in idle
            self.steps.append({'label':'P-G2-reused','source':'completed original production arm before logprob availability probe'})
            self.model_sequence(module.read(self.out/'P-G2-counts.json'))
        except Exception as error:
            self.error=repr(error);self.state('failed',error=self.error)
        finally:self.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','tools','fixtures','dataset'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,module.interrupted);signal.signal(signal.SIGINT,module.interrupted)
    GeneratedGate(args).execute()


if __name__=='__main__':main()
