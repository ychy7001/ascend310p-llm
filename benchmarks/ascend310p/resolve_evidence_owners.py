#!/usr/bin/env python3
"""仅解析相邻N动作＋明确“负责人分别为N姓名”的有限模式；不调用模型。"""
import argparse,copy,json,re
from pathlib import Path
from check_evidence_protocol import validate,oracle,render

def resolve(actions,spans):
    result=copy.deepcopy(actions);changes=[]
    lists=[]
    for i,span in enumerate(spans):
        match=re.fullmatch(r'负责人分别为([\u4e00-\u9fff·、]+)',span['quote'])
        if match:lists.append((i,span,match.group(1).split('、')))
    if len(lists)!=1:return result,changes,'No unique explicit ordered-owner list; leave for review'
    i,span,names=lists[0];n=len(names)
    if n<2 or i<n or len(set(names))!=n or any(not 2<=len(name)<=12 for name in names):return result,changes,'Unsupported owner list'
    task_spans=spans[i-n:i];ids=[s['id'] for s in task_spans];by_task={row['task_evidence_id']:row for row in result}
    if len(by_task)!=len(result) or set(by_task)!=set(ids):return result,changes,'Action coverage not exactly the adjacent ordered clauses'
    if any(span['id'] not in by_task[k]['evidence_ids'] for k in ids):return result,changes,'Owner list not cited by every action'
    for task,name in zip(ids,names):
        row=by_task[task]
        if row['owner']!=name:changes.append({'task_evidence_id':task,'before':row['owner'],'after':name,'basis':span['id']})
        row['owner']=name;row['assignment']='已分配'
    return result,changes,'Applied restricted adjacent ordered-list rule; other semantic relations remain outside this resolver'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    fixtures=json.loads((a.run/'frozen-fixtures.json').read_text());cases={x['id']:x for x in fixtures['cases']};case=cases[4]
    # Positive test uses oracle only as test input; resolver reads source text, never expected fields.
    base=copy.deepcopy(case['expected']);base[0]['owner'],base[1]['owner']=base[1]['owner'],base[0]['owner'];fixed,changes,_=resolve(base,case['spans']);assert oracle(fixed,case['expected']) and len(changes)==2
    # Refuse missing actions, repeated actions, missing citations and non-adjacent clauses.
    for mode in ('missing','duplicate','uncited','nonadjacent'):
        actions=copy.deepcopy(base);spans=copy.deepcopy(case['spans'])
        if mode=='missing':actions.pop()
        elif mode=='duplicate':actions.append(copy.deepcopy(actions[0]))
        elif mode=='uncited':actions[0]['evidence_ids'].remove('s4')
        else:spans.insert(3,{'id':'inserted','start':0,'end':0,'quote':'背景说明'})
        out,changes,reason=resolve(actions,spans);assert not changes,mode
    rows=[]
    with (a.out/'derived.jsonl').open('x') as f:
        for row in map(json.loads,(a.run/'responses.jsonl').read_text().splitlines()):
            case=cases[row['id']];actions=json.loads(row['response']['choices'][0]['message']['content'])['actions'];fixed,changes,reason=resolve(actions,case['spans']);check=validate(fixed,case['spans']);derived={'id':row['id'],'repeat':row['repeat'],'source_oracle_pass':row['oracle_pass'],'actions':fixed,'changes':changes,'rule':reason,'mechanical':check,'fixture_oracle_pass':oracle(fixed,case['expected']),'rendered':render(fixed,case['spans']) if check['mechanical_pass'] else None};rows.append(derived);f.write(json.dumps(derived,ensure_ascii=False)+'\n')
    summary={'source_model_requests':len(rows),'additional_model_requests':0,'source_oracle_pass':sum(x['source_oracle_pass'] for x in rows),'derived_fixture_oracle_pass':sum(x['fixture_oracle_pass'] for x in rows),'changed_requests':sum(bool(x['changes']) for x in rows),'scope':'Same15 outputs, deterministic restricted-pattern derivation. 3 fixtures only; not general business acceptance or new15 model samples. Other relations/completeness still need validation.','production_adoption':False}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
