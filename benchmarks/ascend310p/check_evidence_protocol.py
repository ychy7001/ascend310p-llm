#!/usr/bin/env python3
"""三会议证据协议原型：原文片段引用、保守状态、模板渲染及独立fixture评分。"""
import argparse,copy,hashlib,json,re,time,urllib.request
from pathlib import Path


def clauses(text):
    rows=[]
    for match in re.finditer(r'[^，；。]+',text):
        rows.append({'id':f's{len(rows)}','start':match.start(),'end':match.end(),'quote':match.group()})
    return rows


def validate(actions, spans):
    by_id={s['id']:s for s in spans};issues=[];review=[];seen=set()
    for row in actions:
        task=row['task_evidence_id'];ids=row['evidence_ids']
        if task in seen:issues.append('duplicate_task:'+task)
        seen.add(task)
        if task not in by_id or task not in ids or any(k not in by_id for k in ids):
            issues.append('invalid_evidence:'+task);continue
        text='；'.join(by_id[k]['quote'] for k in ids);task_text=by_id[task]['quote']
        if row['owner']!='未明确' and row['owner'] not in text:issues.append('unsupported_owner:'+task)
        if row['due_date']!='未明确' and row['due_date'] not in text:issues.append('unsupported_due:'+task)
        if row['due_date']=='未明确' and re.search(r'(?:每|下|本)?周[一二三四五六日天](?:前)?|当天|\d+月\d+日',task_text):issues.append('omitted_explicit_time:'+task)
        if row['completion']=='已确认完成' and not any(re.search(r'已(?:完成|执行|回复|停用)',by_id[k]['quote']) and not re.search(r'是否|没有|并未|尚未|未曾|未完成',by_id[k]['quote']) for k in ids):issues.append('unsupported_completion:'+task)
        if row['completion']=='已取消' and not any('已取消' in by_id[k]['quote'] and not re.search(r'是否|没有|未曾',by_id[k]['quote']) for k in ids):issues.append('unsupported_cancellation:'+task)
        if any('分别为' in by_id[k]['quote'] for k in ids):review.append('ordered_owner_relation:'+task)
        if row['owner']=='未明确':review.append('owner_completeness:'+task)
        if (row['assignment']=='已分配')!=(row['owner']!='未明确'):issues.append('assignment_owner_conflict:'+task)
    return {'mechanical_pass':not issues,'issues':issues,'requires_relation_review':review,
            'scope':'Quote/field support and simple explicit-time checks only; no general semantic relation/completeness proof'}


def oracle(actions,expected):
    keys=('task_evidence_id','owner','due_date','assignment','completion')
    return sorted(tuple(x[k] for k in keys) for x in actions)==sorted(tuple(x[k] for k in keys) for x in expected)


def render(actions,spans):
    by_id={s['id']:s for s in spans}
    return '\n'.join(f"- 原文任务：{by_id[x['task_evidence_id']]['quote']}；负责人：{x['owner']}；时间：{x['due_date']}；分配：{x['assignment']}；完成情况：{x['completion']}。" for x in actions)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--base-url');p.add_argument('--model');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();a.out.mkdir(exist_ok=False)
    specs={4:[('s1','周明','当天',['s1','s4']),('s2','刘洋','周三',['s2','s4']),('s3','何丽','未明确',['s3','s4'])],5:[('s1','IT管理员','当天',['s1']),('s2','所有部门','下周一前',['s2']),('s3','合规负责人','未明确',['s3'])],18:[('s0','销售','每周一',['s0']),('s1','产品','周二',['s1']),('s2','研发','周三',['s2']),('s3','未明确','未明确',['s3'])]}
    cases=[]
    for source in map(json.loads,a.fixtures.read_text().splitlines()):
        if source['id'] not in specs:continue
        transcript=source['messages'][1]['content'].split('\n',1)[1];spans=clauses(transcript)
        assert all(transcript[s['start']:s['end']]==s['quote'] for s in spans)
        expected=[dict(task_evidence_id=t,owner=o,due_date=d,assignment='已分配' if o!='未明确' else '未明确',completion='未确认',evidence_ids=e) for t,o,d,e in specs[source['id']]]
        cases.append({'id':source['id'],'transcript':transcript,'spans':spans,'expected':expected})
    assert len(cases)==3
    frozen={'protocol':'evidence-action-v1','scope':'3 source fixtures x5 repeats; expected mappings held outside model request. Generic checks do not prove ordered relations; fixture oracle separate. No production adoption.','fixture_sha256':hashlib.sha256(a.fixtures.read_bytes()).hexdigest(),'cases':cases}
    (a.out/'frozen-fixtures.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
    tests=[];base=cases[0];valid=copy.deepcopy(base['expected']);assert oracle(valid,base['expected']);assert validate(valid,base['spans'])['mechanical_pass']
    for name,change in [('swapped_owners',lambda v:(v[0].update(owner='刘洋'),v[1].update(owner='周明'))),('fake_evidence',lambda v:v[0].update(evidence_ids=['s999'])),('invented_owner',lambda v:v[0].update(owner='张三')),('invented_date',lambda v:v[0].update(due_date='本周内')),('invented_completion',lambda v:v[0].update(completion='已确认完成')),('missing_action',lambda v:v.pop())]:
        v=copy.deepcopy(valid);change(v);check=validate(v,base['spans']);passed=oracle(v,base['expected']);assert not (passed and check['mechanical_pass'])
        if name not in ('swapped_owners','missing_action'):assert not check['mechanical_pass']
        tests.append({'mutation':name,'mechanical':check,'fixture_oracle':passed})
    (a.out/'validator-mutations.json').write_text(json.dumps(tests,ensure_ascii=False,indent=2))
    if a.prepare_only:return
    assert a.base_url and a.model
    rows=[]
    with (a.out/'responses.jsonl').open('x') as stream:
        for repeat in range(5):
            for case in cases:
                ids=[s['id'] for s in case['spans']]
                schema={'type':'object','properties':{'actions':{'type':'array','minItems':1,'maxItems':len(ids),'items':{'type':'object','properties':{'task_evidence_id':{'type':'string','enum':ids},'owner':{'type':'string'},'due_date':{'type':'string'},'assignment':{'type':'string','enum':['已分配','未明确']},'completion':{'type':'string','enum':['已确认完成','未确认','已取消']},'evidence_ids':{'type':'array','minItems':1,'items':{'type':'string','enum':ids}}},'required':['task_evidence_id','owner','due_date','assignment','completion','evidence_ids'],'additionalProperties':False}}},'required':['actions'],'additionalProperties':False}
                instruction='从原文片段中提取执行动作及常规流程，问题描述和负责人名单不单列为动作。task_evidence_id引用动作所在片段，evidence_ids列出支持该动作各字段的全部片段。只引用ID，不改写原文。owner保留原姓名/部门/角色，无依据写未明确；due_date逐字保留原时间，无依据写未明确。分配和完成分别判断：明确负责人写已分配，否则未明确；只有明确确认完成才写已确认完成，明确已取消才写已取消，其他写未确认。“当天”不证明已完成。按顺序解析“分别为”的对应关系。仅输出符合schema的JSON。原文片段：\n'+json.dumps(case['spans'],ensure_ascii=False)
                payload={'model':a.model,'messages':[{'role':'user','content':instruction}],'temperature':0,'seed':0,'max_tokens':1536,'cache_salt':f'evidence-v1-r{repeat}-case{case["id"]}','chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'evidence_actions','strict':True,'schema':schema}}}
                row={'id':case['id'],'repeat':repeat,'request':payload,'oracle_pass':False}
                try:
                    req=urllib.request.Request(a.base_url+'/v1/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=900) as res:response=json.load(res)
                    row['response']=response;actions=json.loads(response['choices'][0]['message']['content'])['actions'];row['validation']=validate(actions,case['spans']);row['oracle_pass']=response['choices'][0]['finish_reason']=='stop' and oracle(actions,case['expected']);row['rendered']=render(actions,case['spans']) if row['validation']['mechanical_pass'] else None
                except Exception as exc:row['error']=repr(exc)
                rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();print(json.dumps({k:row[k] for k in ('id','repeat','oracle_pass')}),flush=True)
    (a.out/'summary.json').write_text(json.dumps({'requests':len(rows),'fixture_oracle_pass':sum(x['oracle_pass'] for x in rows),'mechanical_pass':sum(x.get('validation',{}).get('mechanical_pass',False) for x in rows),'relation_review_requests':sum(bool(x.get('validation',{}).get('requires_relation_review')) for x in rows),'per_case':{i:{'passed':sum(x['oracle_pass'] for x in rows if x['id']==i),'total':5} for i in specs},'scope':frozen['scope'],'production_adoption':False},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
