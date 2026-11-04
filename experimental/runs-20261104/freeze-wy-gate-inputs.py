#!/usr/bin/env python3
"""CPU 冻结35条 G1 token 输入和预先指定的数值规则，不暴露 NPU。"""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                     separators=(',',':')).encode()).hexdigest()


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(exist_ok=False)
tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
replay_path=a.root/'production-w8a8/replay-prompts.json'
replay=json.loads(replay_path.read_text());cases=[]
for i in range(26):
    key=f'chinese_company_annual_meeting_planning:0:{i}'
    cases.append({'id':f'replay-{i:02d}','source':'frozen business-shape synthetic replay, unchanged',
                  'token_ids':replay[key]})
fpath=a.root/'action-state-fixtures-v1.json'
fixtures=json.loads(fpath.read_text())
for case in fixtures['cases']:
    if not case['id'].startswith('long-'):continue
    ids=tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],
                                     tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    assert len(ids)==case['token_info']['count']
    assert digest(ids)==case['token_info']['token_ids_sha256']
    cases.append({'id':case['id'],'source':'existing synthetic boundary fixture','token_ids':ids})
for length in (16384,32768):
    for variant in range(2):
        text='请根据以下会议记录整理决策与行动项，保留人员、金额和日期。\n'
        text+='\n'.join(f'第{i}项讨论：负责人{["李明","王芳","张强","陈洁"][(i+variant)%4]}汇报研发进度为{(i*17+variant)%100}%，计划11月{1+i%28}日完成，预算{i%900+1}万元。'
                        for i in range(1600))
        ids=tokenizer.encode(text,add_special_tokens=False)
        assert len(ids)>=length
        cases.append({'id':f'meeting-{length}-{variant}',
                      'source':'additional synthetic meeting prefix','token_ids':ids[:length]})
assert len(cases)==35
for case in cases:
    case.update(tokens=len(case['token_ids']),token_ids_sha256=digest(case['token_ids']))
inputs={'protocol':'wy921-model-gate-v2','cases':cases,
        'sources_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [replay_path,fpath]},
        'tokenizer_sha256':hashlib.sha256((a.model/'tokenizer.json').read_bytes()).hexdigest()}
(a.out/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
rule={'protocol':'wy921-model-gate-v2','multiplier':1.5,'absolute_floor':0,
      'input_sha256':hashlib.sha256((a.out/'inputs.json').read_bytes()).hexdigest(),
      'generation':{'temperature':0,'seed':0,'max_tokens':256,'ignore_eos':True,'top_k_logprobs':20},
      'ordering':'P G2; A G1+G2; A_prime G1; freeze concrete thresholds; B G1+G2; only if both gates pass P-A-B-A performance',
      'metrics':['top1_disagreement_rate','mean_folded_topk_kl','max_folded_topk_kl',
                 'max_absolute_target_logprob_difference','mean_top1_divergent_suffix_fraction',
                 'max_top1_divergent_suffix_fraction','mean_greedy_divergent_suffix_fraction',
                 'max_greedy_divergent_suffix_fraction'],
      'rule':'For every metric BvsA <= 1.5 * A_prime_vs_A; no absolute floor; all metrics must pass',
      'first_divergence':'0-based raw indices retained per case; compare mean and max (256-first_divergence)/256, or zero when identical',
      'KL':'Same partition of shared explicit top-k token IDs plus OTHER mass; KL(A||candidate); a lower bound, not full-vocabulary KL',
      'prompt_logprobs_probe':'Probe longest frozen input plus256 tokens at A, before reference capture. If runtime explicitly rejects prompt logprobs, freeze generated mode, compare only identical generated conditioning prefixes including first divergence; do not modify engine.',
      'cold':'Prefix on; distinct frozen salts per request; require zero prefix-cache hits for all G1 captures',
      'G2':'For each of GSM64,13 fields C1,13 fields C3,8 short before,8 short after: abs(B-A)<=1 and abs(B-P)<=1',
      'noise':'A_prime is a fresh process with the same frozen A manifest; concrete numeric thresholds must be written and SHA recorded before B starts'}
(a.out/'threshold-rule.json').write_text(json.dumps(rule,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'cases':len(cases),'input_sha256':rule['input_sha256'],
                  'threshold_rule_sha256':hashlib.sha256((a.out/'threshold-rule.json').read_bytes()).hexdigest(),
                  'lengths':[case['tokens'] for case in cases]}))
