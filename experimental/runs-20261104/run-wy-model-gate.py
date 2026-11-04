#!/usr/bin/env python3
"""本轮 P/A/A′/B 模型门禁；双门禁通过才按 P-A-B-A 测性能；finally恢复。"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

from bench_server_phases import prefix_snapshot
from check_action_state_regression import post
from compare_token_logprobs import adjudicate, compare, digest

IMAGE='d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a'
MODEL='Qwen3.6-35B-A3B-w8a8'
BASE='http://127.0.0.1:18126'
SERVICE='llm-backend-primary.service'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def read(path):return json.loads(path.read_text())


def generated_positions(choice):
    ids=choice['token_ids'];lp=choice['logprobs']
    assert len(ids)==256 and len(lp['top_logprobs'])==256
    positions=[]
    for token,actual,top in zip(ids,lp['token_logprobs'],lp['top_logprobs'],strict=True):
        assert all(key.startswith('token_id:') for key in top),list(top)[:2]
        values={key.removeprefix('token_id:'):value for key,value in top.items()}
        values[str(token)]=actual
        positions.append({'top1':token,'logprobs':values})
    return positions


class Gate:
    output_name='wy921-model-gate-v2'

    def __init__(self,args):
        self.root=args.root.resolve();self.out=self.root/self.output_name
        self.tools=args.tools.resolve();self.fixtures=args.fixtures.resolve()
        self.dataset=args.dataset.resolve();self.container=None;self.paused=False
        self.steps=[];self.started=time.monotonic();self.mode=None;self.error=None
        assert not (self.out/'status.json').exists()
        self.inputs=read(self.out/'inputs.json');self.rule=read(self.out/'threshold-rule.json')
        assert sha(self.out/'inputs.json')==self.rule['input_sha256']
        self.manifest=read(self.root/'production-manifest.json')
        self.state('preparing')

    def state(self,value,**kwargs):
        d={'state':value,'mode':self.mode,'container':self.container,'steps':self.steps,
           'elapsed_s':time.monotonic()-self.started,**kwargs}
        tmp=self.out/'status.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(self.out/'status.json')

    def run(self,label,cmd):
        self.state('running',current=label)
        write(self.out/(label+'-command.json'),cmd);start=time.monotonic()
        with (self.out/(label+'.log')).open('x') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        self.steps.append({'label':label,'returncode':result.returncode,'elapsed_s':time.monotonic()-start})
        if result.returncode:raise RuntimeError(f'{label} failed: {result.returncode}; see saved log')
        if time.monotonic()-self.started>86400:raise RuntimeError('User one-day timebox reached')

    def health(self,base=BASE):
        try:
            with urllib.request.urlopen(base+'/health',timeout=5) as response:return response.status==200
        except urllib.error.URLError:return False

    def wait_health(self,base=BASE):
        # Existing archived startup polling budget; no inference request retry.
        for _ in range(240):
            if self.health(base):return
            if self.container:
                status=subprocess.check_output(['podman','inspect','--format','{{.State.Status}}',self.container],text=True).strip()
                if status=='exited':raise RuntimeError('Model exited during startup')
            time.sleep(5)
        raise RuntimeError('Existing model startup budget exhausted')

    def stop(self,label):
        if self.container is None:return
        name=self.container
        subprocess.run(['podman','stop','--time','30',name],check=True,capture_output=True)
        with (self.out/(label+'-server.log')).open('x') as log:
            subprocess.run(['podman','logs',name],stdout=log,stderr=subprocess.STDOUT,check=True)
        detail=json.loads(subprocess.check_output(['podman','inspect',name],text=True))
        write(self.out/(label+'-container-inspect.json'),detail)
        assert detail[0]['State']['Status']=='exited'
        self.container=None

    def launch(self,label,arm):
        self.state('launching',current=label)
        directory=self.out/label;directory.mkdir()
        manifest=copy.deepcopy(self.manifest);command=manifest['command']
        assert command[command.index('--max-num-seqs')+1]=='4'
        assert json.loads(command[command.index('--compilation-config')+1])['cudagraph_capture_sizes']==[1,2]
        command[command.index('--host')+1]='127.0.0.1';command[command.index('--port')+1]='18126'
        model_mount=command[command.index('serve')+1]
        script=self.tools/'start-production-replica.sh'
        if arm!='P':
            manifest['image']=IMAGE;manifest['env'].pop('ASCEND_CUSTOM_PATH',None)
            manifest['env']['WY_GATE_ARM']=arm
            command.extend(['--worker-cls','wy_gate_worker.WYGateWorker'])
            script=self.tools/'start-wy-gate.sh'
        write(directory/'manifest.json',manifest);(directory/'cache').mkdir()
        self.container='wy921-model-'+label.lower().replace('_','-')+'-'+str(time.time_ns())
        cmd=['podman','run','-d','--pull=never','--name',self.container,'--network','host',
             '--group-add','keep-groups','--ipc','shareable','--shm-size','32g']
        if arm!='P':cmd+=['--env','PYTHONPATH=','--env','QWEN36_C8_ENABLED=0']
        for device in ('davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc'):
            cmd+=['--device','/dev/'+device]
        mounts=[(self.root/'production-w8a8/model',model_mount,'ro'),
                (directory,'/experiment','rw'),(directory/'cache','/root/.cache','rw'),
                (script,'/workspace/start_vllm_ascend.sh','ro'),
                (directory/'manifest.json','/workspace/backend-primary.manifest.json','ro')]
        if arm!='P':
            mounts.extend([(self.root/'upstream-gdn-pair921-v1','/gdn-pair','ro'),
                           (self.root/'native-wy921-v1','/native-wy','ro'),(self.tools,'/controltools','ro')])
        for path in ('/usr/local/bin/npu-smi','/etc/ascend_install.info',
                     '/usr/local/Ascend/driver/version.info','/usr/local/Ascend/driver/lib64','/usr/local/dcmi'):
            mounts.append((path,path,'ro'))
        for src,dest,mode in mounts:cmd+=['-v',f'{src}:{dest}:{mode}']
        if arm!='P':cmd+=['--entrypoint','/bin/bash']
        cmd+=[manifest['image']]
        cmd+=['/workspace/start_vllm_ascend.sh'] if arm!='P' else ['bash','/workspace/start_vllm_ascend.sh']
        self.run(label+'-launch',cmd);self.state('waiting_health',current=label);self.wait_health()

    def g2(self,label):
        common=['--base-url',BASE,'--model',MODEL]
        for phase in ('before','main','after'):
            if phase=='main':
                self.run(label+'-gsm64',['python3',str(self.tools/'eval_gsm8k_http.py'),*common,
                         '--dataset',str(self.dataset),'--limit','64','--seed','310','--max-tokens','512',
                         '--label',label,'--out',str(self.out/(label+'-gsm64.jsonl'))])
                self.run(label+'-fields',['python3',str(self.tools/'check_action_state_regression.py'),*common,
                         '--fixtures',str(self.root/'action-state-fixtures-v1.json'),
                         '--out',str(self.out/(label+'-fields.jsonl'))])
            else:
                self.run(label+'-short-'+phase,['python3',str(self.tools/'check_cache_request_reuse.py'),*common,
                         '--out',str(self.out/(label+'-short-'+phase+'.json'))])
        counts={}
        gsm=[json.loads(x) for x in (self.out/(label+'-gsm64.jsonl')).read_text().splitlines()]
        counts['GSM64']=gsm[-1]['correct']
        fields=[json.loads(x) for x in (self.out/(label+'-fields.jsonl')).read_text().splitlines()]
        requests=[r for r in fields if r['kind']=='request']
        if any('error_type' in r for r in requests):raise RuntimeError('G2 fields transport/format error; inspect raw records')
        for c in (1,3):
            rows=[r for r in requests if r['concurrency_limit']==c];assert len(rows)==13
            counts['fields_C'+str(c)]=sum(r['correct'] for r in rows)
        for phase in ('before','after'):
            short=read(self.out/(label+'-short-'+phase+'.json'))
            counts['short_'+phase]=short['strict_matches']
            if not short['reuse']['identical_text'] or short['reuse']['canary_leaked']:
                raise RuntimeError('G2 A/B/A reuse integrity failed')
        write(self.out/(label+'-G2-counts.json'),counts)
        return counts

    def probe(self):
        self.state('probing_prompt_logprobs',current='A')
        case=max(self.inputs['cases'],key=lambda c:c['tokens'])
        payload={'model':MODEL,'prompt':case['token_ids']+[case['token_ids'][-1]]*256,
                 'temperature':0,'seed':0,'max_tokens':1,'prompt_logprobs':20,
                 'return_token_ids':True,'cache_salt':'wy921-v2-prompt-logprobs-probe'}
        write(self.out/'prompt-logprobs-probe-request.json',payload)
        try:
            response=post(BASE,'/v1/completions',payload)
            write(self.out/'prompt-logprobs-probe-response.json',response)
            probs=response['choices'][0].get('prompt_logprobs')
            if probs is None:
                self.mode='generated'
                reason='Runtime returned no prompt_logprobs for explicit request'
            else:
                assert len(probs)==len(payload['prompt'])
                self.mode='teacher_forcing';reason='Longest prompt+256 probe returned per-token prompt_logprobs'
        except urllib.error.HTTPError as error:
            detail=error.read().decode()
            write(self.out/'prompt-logprobs-probe-error.json',{'status':error.code,'body':detail})
            text=detail.lower()
            if 'logprob' not in text or not any(word in text for word in ('support','implement','disabled')):
                raise
            if not self.health():raise RuntimeError('Prompt-logprob probe stopped engine')
            self.mode='generated';reason=detail
        write(self.out/'observation-mode.json',{'mode':self.mode,'reason':reason,'frozen_before_B':True})

    def g1(self,label,reference=None):
        self.state('collecting_G1',current=label)
        raw=self.out/(label+'-G1-raw');raw.mkdir()
        record={'mode':self.mode,'input_sha256':self.rule['input_sha256'],'cases':[]}
        before=prefix_snapshot(BASE)
        for index,case in enumerate(self.inputs['cases']):
            self.state('collecting_G1',current=label,case=case['id'],completed=index,total=len(self.inputs['cases']))
            payload={'model':MODEL,'prompt':case['token_ids'],'temperature':0,'seed':0,
                     'max_tokens':256,'ignore_eos':True,'logprobs':20,
                     'return_token_ids':True,'return_tokens_as_token_ids':True,
                     'cache_salt':'wy921-v2-greedy-'+case['id']}
            response=post(BASE,'/v1/completions',payload)
            with gzip.open(raw/(case['id']+'-greedy.json.gz'),'wt') as stream:json.dump(response,stream)
            choice=response['choices'][0];ids=choice['token_ids'];assert len(ids)==256
            assert response['usage']['prompt_tokens']==case['tokens']
            row={'id':case['id'],'greedy_ids':ids}
            if reference is None:targets=ids
            else:
                assert reference['cases'][index]['id']==case['id']
                targets=reference['cases'][index]['greedy_ids']
            if self.mode=='teacher_forcing':
                teacher={'model':MODEL,'prompt':case['token_ids']+targets,'temperature':0,'seed':0,
                         'max_tokens':1,'prompt_logprobs':20,'return_token_ids':True,
                         'cache_salt':'wy921-v2-teacher-'+case['id']}
                response=post(BASE,'/v1/completions',teacher)
                with gzip.open(raw/(case['id']+'-teacher.json.gz'),'wt') as stream:json.dump(response,stream)
                probs=response['choices'][0]['prompt_logprobs'];assert len(probs)==case['tokens']+256
                positions=[]
                for item in probs[case['tokens']:]:
                    assert item
                    top1=min(item,key=lambda k:(-item[k]['logprob'],int(k)))
                    positions.append({'top1':int(top1),'logprobs':{str(k):v['logprob'] for k,v in item.items()}})
                row['positions']=positions
            else:row['positions']=generated_positions(choice)
            record['cases'].append(row)
            print(json.dumps({'arm':label,'case':case['id'],'G1_collected':len(record['cases'])}),flush=True)
        after=prefix_snapshot(BASE);delta={key:after[key]-before[key] for key in before}
        assert delta['vllm:prefix_cache_hits_total']==0,delta
        record['prefix_cache_delta']=delta
        sequences=[c['greedy_ids'] for c in (record if reference is None else reference)['cases']]
        record['reference_sequence_sha256']=digest(sequences)
        write(self.out/(label+'-G1.json'),record)
        return record

    def performance(self,label):
        common=['--base-url',BASE,'--model',MODEL]
        for length in (1024,4096,16384,32768):
            name=label+'-prefill-'+str(length)
            self.run(name,['python3',str(self.tools/'bench_server_phases.py'),*common,
                     '--prompt-tokens',str(length),'--output-tokens','256','--repeats','3',
                     '--cache-salt-prefix','wy921-v2-perf-'+str(length),'--out',str(self.out/(name+'.jsonl'))])
        for c in (1,3,4):
            name=label+'-stream-C'+str(c)
            self.run(name,['python3',str(self.tools/'bench_http_stream.py'),*common,'--prompt-tokens','1024',
                     '--output-tokens','256','--concurrency',str(c),'--repeats','3','--seed','0',
                     '--cache-salt-prefix','wy921-v2-stream-'+str(c),'--out',str(self.out/(name+'.jsonl'))])
        prompts=self.out/(label+'-replay-prompts.json')
        shutil.copy2(self.root/'production-w8a8/replay-prompts.json',prompts)
        self.run(label+'-replay',['python3',str(self.tools/'replay_stage_workload.py'),
                 '--workload',str(self.fixtures/'a1-310p-tune.workload.json'),
                 '--recording','chinese_company_annual_meeting_planning','--urls',BASE,'--model',MODEL,
                 '--prompts',str(prompts),'--out',str(self.out/(label+'-replay.json'))])
        replay=read(self.out/(label+'-replay.json'));assert len(replay['calls'])==26 and replay['cold_cache_verified']

    def model_sequence(self,p):
        self.launch('A','A')
        if self.mode is None:self.probe()
        a=self.g1('A');ga=self.g2('A');self.stop('A')
        self.launch('A_prime','A');ap=self.g1('A_prime',a);self.stop('A_prime')
        noise=compare(a,ap);write(self.out/'A_prime-vs-A.json',noise)
        self.candidate_sequence(p,a,ga,noise)

    def candidate_sequence(self,p,a,ga,noise):
        thresholds={'rule_sha256':sha(self.out/'threshold-rule.json'),
                    'A_reference_sha256':sha(self.out/'A-G1.json'),
                    'noise_sha256':sha(self.out/'A_prime-vs-A.json'),
                    'thresholds':{key:1.5*value for key,value in noise['metrics'].items()},
                    'frozen_at_unix_ns':time.time_ns(),'B_started':False}
        write(self.out/'concrete-thresholds.json',thresholds)
        threshold_sha=sha(self.out/'concrete-thresholds.json')
        self.state('thresholds_frozen_before_B',concrete_thresholds_sha256=threshold_sha)
        print(json.dumps({'thresholds_frozen_before_B':threshold_sha,'noise':noise['metrics']}),flush=True)
        self.launch('B','B');b=self.g1('B',a);gb=self.g2('B');self.stop('B')
        paired=compare(a,b);write(self.out/'B-vs-A.json',paired)
        numerical=adjudicate(noise,paired,self.rule)
        numerical['concrete_thresholds_sha256']=threshold_sha
        assert {key:row['threshold'] for key,row in numerical['checks'].items()}==thresholds['thresholds']
        write(self.out/'G1-adjudication.json',numerical)
        tasks={'counts':{'P':p,'A':ga,'B':gb},'checks':{}}
        for key in p:
            tasks['checks'][key]={'B_vs_A_difference':abs(gb[key]-ga[key]),
                                 'B_vs_P_difference':abs(gb[key]-p[key]),
                                 'passed':abs(gb[key]-ga[key])<=1 and abs(gb[key]-p[key])<=1}
        tasks['passed']=all(row['passed'] for row in tasks['checks'].values())
        write(self.out/'G2-adjudication.json',tasks)
        if numerical['passed'] and tasks['passed']:
            for label,arm in [('perf-P','P'),('perf-A1','A'),('perf-B','B'),('perf-A2','A')]:
                self.launch(label,arm);self.performance(label);self.stop(label)
            self.state('gates_passed_performance_complete')
        else:self.state('gate_rejected_performance_not_run',G1_passed=numerical['passed'],G2_passed=tasks['passed'])

    def execute(self):
        try:
            assert self.health('http://127.0.0.1:18081')
            assert read(self.root/'native-wy921-gate-v2/status.json')['state']=='passed_operator_gate'
            assert read(self.root/'upstream-gdn-pair921-v1/status.json')['state']=='passed_operator_validation'
            assert sha(self.dataset)=='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
            write(self.out/'tool-freeze.json',{p.name:sha(p) for p in self.tools.iterdir() if p.is_file()})
            self.paused=True
            subprocess.run(['sudo','-n','systemctl','stop',SERVICE],check=True)
            idle=subprocess.check_output(['npu-smi','info'],text=True)
            (self.out/'idle-before.txt').write_text(idle)
            assert 'No running processes found' in idle
            self.launch('P','P');p=self.g2('P');self.stop('P')
            self.model_sequence(p)
        except Exception as error:
            self.error=repr(error);self.state('failed',error=self.error)
        finally:
            self.finish()

    def finish(self):
        try:
            if self.container:self.stop('emergency')
        finally:
            if self.paused:
                self.state('restoring_original_service',primary_error=self.error)
                with (self.out/'restore-service.log').open('x') as log:
                    subprocess.run(['sudo','-n','systemctl','reset-failed',SERVICE],stdout=log,stderr=subprocess.STDOUT)
                    result=subprocess.run(['sudo','-n','systemctl','start',SERVICE],stdout=log,stderr=subprocess.STDOUT)
                self.wait_health('http://127.0.0.1:18081')
                restored={'start_returncode':result.returncode,'health_200':self.health('http://127.0.0.1:18081'),
                          'active_state':subprocess.check_output(['systemctl','is-active',SERVICE],text=True).strip()}
                write(self.out/'service-restoration.json',restored)
                assert result.returncode==0 and restored['health_200'] and restored['active_state']=='active'
            g1=self.out/'G1-adjudication.json';g2=self.out/'G2-adjudication.json'
            self.state('failed' if self.error else 'complete',error=self.error,
                       G1_passed=read(g1)['passed'] if g1.exists() else None,
                       G2_passed=read(g2)['passed'] if g2.exists() else None,
                       performance_complete=(self.out/'perf-A2-replay.json').exists())


def interrupted(signum,frame):raise RuntimeError('Interrupted signal '+str(signum))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','tools','fixtures','dataset'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    Gate(args).execute()


if __name__=='__main__':main()
