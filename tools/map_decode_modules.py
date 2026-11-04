#!/usr/bin/env python3
"""Map compiled FX/module metadata and real tensor descriptors, without tensor data."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re


def module_path(node):
    stack=node.get('module_stack',{})
    if not stack:return None
    value=list(stack.values())[-1]
    match=re.search(r'\("([^\"]+)"',value)
    if not match:return None
    return match.group(1).replace("L['self'].",'language_model.model.')


def reaches_output(nodes):
    names={n['name'] for n in nodes};users=defaultdict(set)
    for node in nodes:
        inputs=set(re.findall(r'\b[A-Za-z_]\w*\b',node['args']+' '+node['kwargs']))&names
        for source in inputs:users[source].add(node['name'])
    outputs={n['name'] for n in nodes if n['op']=='output'}
    memo={}
    def visit(name,visiting=None):
        if name in outputs:return True
        if name in memo:return memo[name]
        visiting=set() if visiting is None else visiting
        if name in visiting:return False
        visiting=visiting|{name}
        memo[name]=any(visit(other,visiting) for other in users[name]);return memo[name]
    return {name:visit(name) for name in names}


def module_group(name):
    if '.shared_expert.gate_up_proj' in name:return 'shared_gate_up'
    if '.shared_expert.down_proj' in name:return 'shared_down'
    if '.linear_attn.in_proj_qkvz' in name:return 'linear_qkvz'
    if '.linear_attn.in_proj_ba' in name:return 'linear_ba'
    if '.linear_attn.out_proj' in name:return 'linear_out_proj'
    if '.self_attn.o_proj' in name:return 'full_attention_o_proj'
    if '.mlp.gate' in name:return 'moe_router'
    return 'other'


def build_ledger(events):
    graphs=[r for r in events if r['kind']=='fx_graph']
    graph=next(g for g in graphs if sum(n['target']=='vllm.unquantized_gemm.default' for n in g['nodes'])==140)
    main=[n for n in graph['nodes'] if n['target']=='vllm.unquantized_gemm.default']
    reduce_nodes=[n for n in graph['nodes'] if n['target'] in ('vllm.all_reduce.default','vllm.maybe_all_reduce_tensor_model_parallel.default')]
    nodes_by_name={n['name']:n for n in graph['nodes']};reach=reaches_output(graph['nodes'])
    shared_nodes={re.search(r'\.layers\.(\d+)\.',module_path(n)).group(1):n['name'] for n in graph['nodes'] if n['target']=='vllm.moe_forward_shared.default'}
    starts=[r for r in events if r['kind']=='forward_start' and r.get('mapping')]
    ledgers=[];reductions=[];validation=[]
    for start in starts:
        end=next(r for r in events if r['kind']=='forward_return' and r['seq']==start['seq'] and r['mono']>=start['mono'])
        rows=[r for r in events if start['mono']<=r['mono']<=end['mono']]
        ledger=[];ri=0;mi=0;reduce_ledger=[]
        for row in rows:
            if row['kind']=='torch_op' and row['op']=='vllm.unquantized_gemm.default':
                node=main[mi];mi+=1
                module=module_path(node)
                if module is None:raise ValueError('Missing module stack for '+node['name'])
                inputs=row['inputs'];weight_name=re.findall(r'\b\w+\b',node['args'])[1]
                expected=nodes_by_name[weight_name].get('value',{})
                if expected and expected.get('shape')!=inputs[1]['shape']:raise ValueError('FX/real weight shape mismatch')
                ledger.append({'module':module,'group':module_group(module),'fx_node':node['name'],'op':row['op'],
                               'input':inputs[0],'weight':inputs[1],'output':row['outputs'],'evidence':'FX nn_module_stack + matched runtime op order and weight shape'})
            elif row['kind']=='module_enter' and row['module'].endswith(('.shared_expert.gate_up_proj','.shared_expert.down_proj')):
                ledger.append({'module':row['module'],'group':module_group(row['module']),
                               'fx_node':shared_nodes[re.search(r'\.layers\.(\d+)\.',row['module']).group(1)],
                               'op':'shared expert module forward','input':row['inputs'][0],'weight':row['parameters']['weight'],
                               'output':None,'evidence':'actual module forward hook nested under moe_forward_shared'})
            elif row['kind']=='torch_op' and row['op'] in ('vllm.all_reduce.default','vllm.maybe_all_reduce_tensor_model_parallel.default'):
                node=reduce_nodes[ri];ri+=1
                if node['target']!=row['op']:raise ValueError('Reduction order mismatch')
                reduce_ledger.append({'ordinal':ri-1,'module':module_path(node),'fx_node':node['name'],'op':row['op'],
                                      'input':row['inputs'][0],'reaches_graph_output':reach[node['name']]})
        for ordinal,row in enumerate(ledger):row['ordinal']=ordinal
        if len(ledger)!=220 or mi!=140 or ri!=80:raise ValueError(('incomplete forward',start['seq'],len(ledger),mi,ri))
        ledgers.append(ledger);reductions.append(reduce_ledger)
        validation.append({'seq':start['seq'],'fp16_projection_nodes':len(ledger),'main_fx_calls':mi,'reductions':ri,
                           'all_reductions_reach_graph_output':all(r['reaches_graph_output'] for r in reduce_ledger)})
    if not ledgers:raise ValueError('No mapped forwards')
    template=[(r['module'],r['weight']['shape'],r['weight']['dtype']) for r in ledgers[0]]
    stable=all([(r['module'],r['weight']['shape'],r['weight']['dtype']) for r in l]==template for l in ledgers)
    if not stable:raise ValueError('Mapped forward order differs')
    return {'mapped_forward_count':len(ledgers),'real_batch':starts[0]['batch'],'dtype':'torch.float16',
            'group_counts':dict(Counter(r['group'] for r in ledgers[0])),
            'forwards':validation,'identical_order_and_weight_shapes':stable,'fp16_ledger':ledgers[0],
            'reduction_ledger':reductions[0],
            'scope':'Real B3 forward descriptors and compiled FX; ACLGraph task IDs are joined separately by stable 220/80-node order, not by module CPU duration.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observation-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();results=[]
    for path in a.observation_dir.glob('*.jsonl'):
        events=[json.loads(l) for l in path.read_text().splitlines()]
        if not any(e['kind']=='fx_graph' for e in events):continue
        result=build_ledger(events);result['rank']=next(e['rank'] for e in events if e['kind']=='execute_start')
        results.append(result)
    if len(results)!=2:raise ValueError('Expected both mapped ranks')
    a.out.write_text(json.dumps(sorted(results,key=lambda r:r['rank']),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('fp16_ledger','reduction_ledger')} for r in results],indent=2))

if __name__=='__main__':main()
