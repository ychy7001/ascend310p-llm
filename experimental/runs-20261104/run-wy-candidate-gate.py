#!/usr/bin/env python3
"""复用完整 A/A′/P 记录，冻结有限的共同 top-k 条件 KL 后只运行 B。"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import signal
import subprocess

spec=importlib.util.spec_from_file_location('wy_model_gate',Path(__file__).with_name('run-wy-model-gate.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CandidateGate(module.Gate):
    output_name='wy921-model-gate-final-v1'

    def __init__(self,args):
        root=args.root.resolve();previous=root/'wy921-model-gate-generated-v1'
        status=module.read(previous/'status.json')
        assert status['state']=='failed' and 'not JSON compliant: inf' in status['error']
        restored=module.read(previous/'service-restoration.json')
        assert restored['health_200'] and restored['active_state']=='active'
        assert not (previous/'B').exists()
        out=root/self.output_name;out.mkdir(exist_ok=False)
        shutil.copy2(previous/'inputs.json',out/'inputs.json')
        rule=module.read(previous/'threshold-rule.json')
        rule['previous_threshold_rule_sha256']=module.sha(previous/'threshold-rule.json')
        rule['metrics']=[name.replace('folded_topk_kl','conditional_topk_kl') for name in rule['metrics']]
        rule['KL']='KL(A||candidate) after separately normalizing on shared explicit top-k token IDs with log-sum-exp; excludes probability mass outside shared support; not full-vocabulary KL and not a full-KL lower bound. No probability floor. All other metrics and 1.5 multiplier unchanged.'
        rule['ordering']='Reuse complete P G2, A G1/G2 and fresh A_prime G1; freeze conditional-top-k method and concrete numeric thresholds; first B G1+G2; only if both pass P-A-B-A performance'
        module.write(out/'threshold-rule.json',rule)
        for arm in ('P','A','A_prime'):
            shutil.copytree(previous/arm,out/arm,ignore=lambda path,names:set(names)&{'cache'})
        for path in previous.iterdir():
            if path.name in {'inputs.json','threshold-rule.json','status.json','A_prime-vs-A.json','tool-freeze.json','idle-before.txt','restore-service.log','service-restoration.json'}:continue
            if path.is_file():shutil.copy2(path,out/path.name)
            elif path.name.endswith('-G1-raw'):shutil.copytree(path,out/path.name)
        shutil.copy2(previous/'status.json',out/'numerical-report-attempt-status.json')
        shutil.copy2(previous/'service-restoration.json',out/'numerical-report-attempt-restoration.json')
        # Preserve the incomplete diagnostic serialization rather than overwriting it.
        shutil.copy2(previous/'A_prime-vs-A.json',out/'numerical-report-attempt-partial.json')
        super().__init__(args);self.mode='generated'
        self.reference=module.read(out/'A-G1.json')
        self.noise=module.compare(self.reference,module.read(out/'A_prime-G1.json'))
        module.write(out/'A_prime-vs-A.json',self.noise)
        self.state('conditional_topk_method_frozen_before_B',threshold_rule_sha256=module.sha(out/'threshold-rule.json'))
        print(module.json.dumps({'threshold_rule_sha256':module.sha(out/'threshold-rule.json'),'noise':self.noise['metrics']}),flush=True)

    def execute(self):
        try:
            assert self.health('http://127.0.0.1:18081')
            module.write(self.out/'tool-freeze.json',{p.name:module.sha(p) for p in self.tools.iterdir() if p.is_file()})
            self.paused=True
            subprocess.run(['sudo','-n','systemctl','stop',module.SERVICE],check=True)
            idle=subprocess.check_output(['npu-smi','info'],text=True)
            (self.out/'idle-before.txt').write_text(idle)
            assert 'No running processes found' in idle
            self.steps.append({'label':'P-A-A_prime-reused','source':'completed captures before first B; only CPU KL aggregation changed'})
            self.candidate_sequence(module.read(self.out/'P-G2-counts.json'),self.reference,
                                    module.read(self.out/'A-G2-counts.json'),self.noise)
        except Exception as error:
            self.error=repr(error);self.state('failed',error=self.error)
        finally:self.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','tools','fixtures','dataset'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,module.interrupted);signal.signal(signal.SIGINT,module.interrupted)
    CandidateGate(args).execute()


if __name__=='__main__':main()
