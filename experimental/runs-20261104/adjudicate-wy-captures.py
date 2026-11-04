#!/usr/bin/env python3
"""离线完成原始记录判定；核对 B 前阈值，删失 logprob 只报保守下界。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from compare_token_logprobs import adjudicate,compare


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);args=parser.parse_args();raw=args.raw
    frozen=read(raw/'concrete-thresholds.json');rule=read(raw/'threshold-rule.json')
    assert sha(raw/'threshold-rule.json')==frozen['rule_sha256']
    assert sha(raw/'A-G1.json')==frozen['A_reference_sha256']
    assert sha(raw/'A_prime-vs-A.json')==frozen['noise_sha256']
    a=read(raw/'A-G1.json');noise=compare(a,read(raw/'A_prime-G1.json'))
    assert noise['metrics']==read(raw/'A_prime-vs-A.json')['metrics']
    assert {name:1.5*value for name,value in noise['metrics'].items()}==frozen['thresholds']
    candidate=compare(a,read(raw/'B-G1.json'));write(raw/'B-vs-A.json',candidate)
    gate=adjudicate(noise,candidate,rule)
    gate.update(concrete_thresholds_sha256=sha(raw/'concrete-thresholds.json'),
                evidence_note='Censored A target tokens outside candidate top-20 yield lower bounds, not zero errors; a lower bound above a frozen threshold is sufficient to reject. A bound below threshold cannot pass.')
    write(raw/'G1-adjudication.json',gate)
    counts={arm:read(raw/(arm+'-G2-counts.json')) for arm in ('P','A','B')}
    tasks={'counts':counts,'checks':{}}
    for name in counts['P']:
        da=abs(counts['B'][name]-counts['A'][name]);dp=abs(counts['B'][name]-counts['P'][name])
        tasks['checks'][name]={'B_vs_A_difference':da,'B_vs_P_difference':dp,'passed':da<=1 and dp<=1}
    tasks['passed']=all(row['passed'] for row in tasks['checks'].values())
    write(raw/'G2-adjudication.json',tasks)
    write(raw/'offline-adjudication.json',{
        'G1_passed':gate['passed'],'G2_passed':tasks['passed'],'thresholds_changed':False,
        'new_NPU_requests':0,'comparison_tool_sha256':sha(Path(__file__).resolve().parents[2]/'tools/compare_token_logprobs.py'),
        'reason':'Capture completed; original online aggregator rejected omitted target logprobs. Conservative censored-error reporting completed offline with the exact same frozen numeric thresholds.'})
    print(json.dumps({'G1_passed':gate['passed'],'G2_passed':tasks['passed'],
                      'positions_compared':candidate['positions_compared'],
                      'censored_targets':candidate['target_logprob_censoring']['count'],
                      'checks':gate['checks']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
