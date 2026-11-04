#!/usr/bin/env python3
"""CPU-only kernel/task accounting over stable head-to-head decode windows."""
import argparse
import bisect
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics
import sys
from decode_trace_utils import (trace_events,relative_timestamp,union_duration,intersection_duration,subtract,kernel_category)
from inference_numeric_metadata import quantiles


def load_kernels(path):
    opener=gzip.open if path.suffix=='.gz' else open
    rows=[];origin=None
    with opener(path,'rt') as stream:
        for row in csv.DictReader(stream):
            raw=(row.get('Start Time(us)') or row.get('Task Start Time(us)')).strip()
            if origin is None:origin=int(float(raw)//1000000)*1000000
            start=relative_timestamp(raw,origin);duration=float(row.get('Duration(us)') or row['Task Duration(us)'])
            rows.append({'start':start,'end':start+duration,'duration':duration,'name':sys.intern(row.get('Name',row.get('Op Name',''))),
                         'category':kernel_category(row),'model':row.get('Model ID'),'task':row.get('Task ID'),
                         'stream':row.get('Stream ID'),'shapes':sys.intern(row.get('Input Shapes','')),
                         'dtypes':sys.intern(row.get('Input Data Types','')),'formats':sys.intern(row.get('Input Formats',''))})
    rows.sort(key=lambda r:r['start'])
    return origin,rows


def stable_window(rows,maximum_steps=96,tail=False):
    heads=[r['start'] for r in rows if r['category']=='lm_head_fp16']
    if len(heads)<66:raise ValueError('Insufficient head boundaries: %d'%len(heads))
    periods=[b-a for a,b in zip(heads,heads[1:])];median=statistics.median(periods)
    # Prefer a contiguous stable window and exclude profiler startup/stop and mapped forwards.
    longest=[];current=[]
    for i,duration in enumerate(periods):
        if .5*median<duration<1.5*median:
            current.append(i)
        else:
            if len(current)>len(longest):longest=current
            current=[]
    if len(current)>len(longest) or tail and len(current)>=64:longest=current
    if len(longest)<64:raise ValueError('No >=64-step stable window')
    count=min(maximum_steps,len(longest));offset=len(longest)-count if tail else (len(longest)-count)//2
    indices=longest[offset:offset+count]
    lo=heads[indices[0]];hi=heads[indices[-1]+1]
    return lo,hi,count,{'captured_head_count':len(heads),'stable_head_period_us':quantiles([periods[i] for i in indices]),
                         'first_selected_head_index':indices[0],'last_selected_head_index':indices[-1],
                         'all_head_period_us':quantiles(periods)}


def analyze(csv_path,trace_path,maximum_steps=96,tail=False):
    origin,all_rows=load_kernels(csv_path);lo,hi,steps,selection=stable_window(all_rows,maximum_steps,tail)
    rows=[r for r in all_rows if r['start']<hi and r['end']>lo]
    categories=defaultdict(list);names=Counter();counts=Counter();shapes=defaultdict(Counter);nodes={}
    for row in rows:
        interval=(max(lo,row['start']),min(hi,row['end']));categories[row['category']].append(interval)
        names[row['name']]+=interval[1]-interval[0];counts[row['name']]+=1
        shapes[row['name']][(row['shapes'],row['dtypes'],row['formats'])]+=1
        if row['category']=='fp16_nz_projection':
            nodes.setdefault((row['model'],row['stream'],row['task']),{'name':row['name'],'shapes':row['shapes'],'dtypes':row['dtypes'],'formats':row['formats']})
    known=[iv for values in categories.values() for iv in values]
    compute=[iv for key,values in categories.items() if not key.startswith('visible_') for iv in values]
    comm=[iv for key,values in categories.items() if key.startswith('visible_') for iv in values]
    span=hi-lo;known_union=union_duration(known);gap=subtract([(lo,hi)],known)
    fp16_lookup={(r['name'],round(r['start'],3)):r for r in rows if r['category']=='fp16_nz_projection'}
    graph_reductions=[]
    hardware=defaultdict(list);task_counts=Counter();model_counts=Counter();raw_samples={};processes={};flows=Counter()
    host_ops=[];host_sync=[];host_annotations=[];host_api=[]
    if trace_path:
        with trace_path.open() as stream:
            for event in trace_events(stream):
                if event.get('ph')=='M' and event.get('name')=='process_name':
                    processes[str(event['pid'])]=event.get('args',{}).get('name','');continue
                if event.get('ph') in ('s','f'):flows[event.get('cat','')]+=1;continue
                if event.get('ph')!='X':continue
                start=relative_timestamp(event['ts'],origin);end=start+float(event.get('dur',0))
                if start>=hi or end<=lo:continue
                interval=(max(lo,start),min(hi,end));name=event.get('name','');args=event.get('args',{})
                if 'Task Type' in args and ('Model Id' in args or name=='MemcopyAsync'):
                    matched=fp16_lookup.get((name,round(start,3)))
                    if matched is not None:
                        matched.update(hardware_model=str(args.get('Model Id')),hardware_task=str(args.get('Task Id')),hardware_stream=str(args.get('Physic Stream Id')))
                    if name=='REDUCE_ASYNC_V2' and args.get('Model Id') not in (None,4294967295):
                        graph_reductions.append({'start':start,'end':end,'model':str(args['Model Id']),'stream':str(args.get('Physic Stream Id')),'task':str(args.get('Task Id'))})
                    hardware[name].append(interval);task_counts[name]+=1;model_counts[(name,str(args.get('Model Id')))]+=1
                    if name in ('REDUCE_ASYNC_V2','MEMCPY_ASYNC','NOTIFY_WAIT','NOTIFY_RECORD','EVENT_WAIT'):
                        raw_samples.setdefault(name,{'name':name,'duration_us':float(event.get('dur',0)),'args':args})
                cat=event.get('cat','')
                if cat=='cpu_op':host_ops.append(interval)
                if 'synchronize' in name.lower():host_sync.append(interval)
                if name.startswith('decode_module::'):host_annotations.append(interval)
                if name.startswith('AscendCL@'):host_api.append(interval)
    graph_fp16_ids={(r.get('hardware_model'),r.get('hardware_stream'),r.get('hardware_task')) for r in fp16_lookup.values() if r.get('hardware_model') not in (None,'4294967295')}
    heads=[r['start'] for r in all_rows if r['category']=='lm_head_fp16']
    reductions_by_step=defaultdict(list)
    for r in sorted(graph_reductions,key=lambda r:r['start']):reductions_by_step[bisect.bisect_right(heads,r['start'])-1].append(r)
    reduction_nodes=[]
    reduction_templates={tuple((r['model'],r['stream'],r['task']) for r in rr) for rr in reductions_by_step.values()}
    if len(reduction_templates)==1 and len(reductions_by_step)==steps:
        template=next(iter(reduction_templates))
        for ordinal,identifier in enumerate(template):
            iv=[(rr[ordinal]['start'],rr[ordinal]['end']) for rr in reductions_by_step.values()]
            reduction_nodes.append({'ordinal':ordinal,'model':identifier[0],'stream':identifier[1],'task':identifier[2],
                                    'count':len(iv),'ms_per_step':sum(b-a for a,b in iv)/steps/1000,
                                    'overlap_compute_ms_per_step':intersection_duration(iv,compute)/steps/1000})
    raw_reduce=hardware.get('REDUCE_ASYNC_V2',[])
    raw_transfer=hardware.get('MEMCPY_ASYNC',[])
    active_other=[iv for name,values in hardware.items() if name not in ('EVENT_WAIT','NOTIFY_WAIT') for iv in values]
    result={'steps':steps,'origin_epoch_us':origin,'window_start_us':lo,'window_end_us':hi,
            'window_ms_per_step':span/steps/1000,'selection':selection,'kernel_rows':len(rows),
            'named_kernel_sum_ms_per_step':sum(sum(b-a for a,b in v) for v in categories.values())/steps/1000,
            'named_kernel_union_ms_per_step':known_union/steps/1000,
            'named_kernel_overlap_ms_per_step':(sum(sum(b-a for a,b in v) for v in categories.values())-known_union)/steps/1000,
            'uncovered_by_named_kernel_ms_per_step':(span-known_union)/steps/1000,
            'categories':{k:{'count':len(v),'sum_ms_per_step':sum(b-a for a,b in v)/steps/1000,
                               'union_ms_per_step':union_duration(v)/steps/1000,'window_fraction':union_duration(v)/span} for k,v in categories.items()},
            'top_kernels':[{'name':n,'count':counts[n],'ms_per_step':v/steps/1000,
                            'shape_samples':[{'shapes':key[0],'dtypes':key[1],'formats':key[2],'count':count} for key,count in shapes[n].most_common(4)]} for n,v in names.most_common(25)],
            'graph_fp16_node_count':len(graph_fp16_ids),
            'fp16_hardware_metadata_matches':sum('hardware_model' in r for r in fp16_lookup.values()),
            'graph_reduction_node_ledger':reduction_nodes,
            'graph_reduction_counts_by_step':dict(Counter(len(v) for v in reductions_by_step.values())),
            'hardware_tasks':{n:{'count':task_counts[n],'sum_ms_per_step':sum(b-a for a,b in iv)/steps/1000,
                                  'union_ms_per_step':union_duration(iv)/steps/1000,
                                  'overlap_compute_ms_per_step':intersection_duration(iv,compute)/steps/1000} for n,iv in hardware.items()},
            'hardware_models':[{'task':k[0],'model':k[1],'count':v} for k,v in model_counts.items()],
            'raw_reduce_outside_compute_ms_per_step':(union_duration(raw_reduce)-intersection_duration(raw_reduce,compute))/steps/1000,
            'raw_transfer_outside_compute_ms_per_step':(union_duration(raw_transfer)-intersection_duration(raw_transfer,compute))/steps/1000,
            'uncovered_by_all_active_hardware_ms_per_step':(span-union_duration(known+active_other))/steps/1000,
            'worker_cpu_op_union_ms_per_step':union_duration(host_ops)/steps/1000,
            'worker_cpu_ops_in_named_kernel_gap_ms_per_step':intersection_duration(host_ops,gap)/steps/1000,
            'host_sync_span_ms_per_step_overlaps_device':union_duration(host_sync)/steps/1000,
            'module_annotation_count':len(host_annotations),'flow_categories':dict(flows),
            'raw_task_samples':raw_samples}
    return result,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--trace',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=96);p.add_argument('--tail',action='store_true');p.add_argument('--ledger',type=Path);p.add_argument('--rank',type=int);a=p.parse_args()
    result,rows=analyze(a.csv,a.trace,a.steps,a.tail)
    if a.ledger:
        from attribute_decode_kernels import attribute
        ledger=next(r for r in json.loads(a.ledger.read_text()) if r['rank']==a.rank)['fp16_ledger']
        result['fp16_module_attribution']=attribute(rows,ledger,result['window_start_us'],result['window_end_us'],result['steps'])
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('top_kernels','hardware_tasks','hardware_models','raw_task_samples')}))

if __name__=='__main__':main()
