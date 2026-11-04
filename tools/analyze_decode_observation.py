#!/usr/bin/env python3
"""Join read-only scheduler and runner observations using FIFO signatures and step IDs.

Stage durations use CPU-observed output completion. They include device/TP waits;
GPU kernel time comes from separate profiles. Async intervals are clipped at the
previous completion so service time is never counted twice.
"""
import argparse
from collections import Counter,defaultdict,deque
import csv
import json
from pathlib import Path
import statistics
from inference_numeric_metadata import quantiles


def signature(requests):
    if isinstance(requests,dict):return tuple(sorted(requests.items()))
    return tuple(sorted((r['id'],r['scheduled']) for r in requests))


def join_steps(events,rank=0):
    schedules=[e for e in events if e['kind']=='schedule' and e['requests']]
    schedules.sort(key=lambda e:e['mono'])
    by_signature=defaultdict(deque)
    for row in schedules:by_signature[signature(row['requests'])].append(row)
    completions={e['step']:e for e in events if e['kind']=='completion' and e.get('step') is not None}
    executes=[e for e in events if e['kind']=='execute_start' and e.get('rank')==rank and e['requests']]
    executes.sort(key=lambda e:e['mono'])
    pids={e['pid'] for e in executes}
    if len(pids)!=1:raise ValueError('Expected exactly one active worker per rank, got %s'%pids)
    pid=next(iter(pids))
    forwards={e['seq']:e for e in events if e.get('pid')==pid and e['kind']=='forward_start' and not e['capturing']}
    returns={e['seq']:e for e in events if e.get('pid')==pid and e['kind']=='execute_return'}
    samples={e['seq']:e for e in events if e.get('pid')==pid and e['kind']=='sample_return'}
    result=[];unmatched=[];previous_end=None
    for execute in executes:
        key=signature(execute['requests'])
        if not by_signature[key]:unmatched.append(execute['seq']);continue
        scheduled=by_signature[key].popleft();completion=completions.get(scheduled['step'])
        forward=forwards.get(execute['seq'])
        if completion is None or forward is None:unmatched.append(execute['seq']);continue
        start=max(execute['mono'],previous_end if previous_end is not None else execute['mono'])
        end=completion['mono']
        if end<start:raise ValueError('Non-monotonic output completion')
        previous_end=end
        prefill=[r for r in scheduled['requests'] if r['phase']=='prefill']
        decode=[r for r in scheduled['requests'] if r['phase']=='decode']
        phase='mixed' if prefill and decode else ('prefill' if prefill else 'decode')
        row={'step':scheduled['step'],'seq':execute['seq'],'start':start,'end':end,'duration_s':end-start,
             'dispatch_start':execute['mono'],'schedule_start':scheduled['start'],
             'phase':phase,'batch':len(scheduled['requests']),'decode_batch':len(decode),'prefill_batch':len(prefill),
             'padded_tokens':forward['padded_tokens'],'graph_hit':forward['graph_hit'],'mode':forward['mode'],
             'contexts':[r['computed'] for r in scheduled['requests']],
             'max_context':max(r['computed'] for r in scheduled['requests']),
             'requests':scheduled['requests'],'completion_requests':completion['requests'],
             'scheduler_cpu_s':scheduled['end']-scheduled['start']+completion['end']-completion['start'],
             'execute_cpu_span_s':returns.get(execute['seq'],{}).get('end',start)-execute['mono'],
             'sample_cpu_span_s':samples.get(execute['seq'],{}).get('end',end)-samples.get(execute['seq'],{}).get('start',end),
             'mapping':forward.get('mapping',False)}
        result.append(row)
    return result,{'scheduler_steps':len(schedules),'worker_steps':len(executes),'joined_steps':len(result),
                   'unmatched_worker_sequences':unmatched,'worker_pid':pid,'rank':rank}


def analyze(events,http):
    steps,validation=join_steps(events)
    begin=next(e['mono'] for e in http if e['kind']=='replay_start')
    finish=next(e['mono'] for e in http if e['kind']=='replay_end')
    replay=[r for r in steps if begin<=r['dispatch_start']<finish]
    phases=Counter();phase_steps=Counter();batch_seconds=Counter();batch_count=Counter();groups=defaultdict(list)
    for row in replay:
        phases[row['phase']]+=row['duration_s'];phase_steps[row['phase']]+=1
        if row['phase']=='decode':
            batch_seconds[row['decode_batch']]+=row['duration_s'];batch_count[row['decode_batch']]+=1
            c=row['max_context'];bucket=next((name for ceiling,name in [(8192,'0-8k'),(16384,'8-16k'),(32768,'16-32k'),(65536,'32-64k'),(float('inf'),'64k+')] if c<ceiling))
            groups[(row['decode_batch'],bucket)].append(row)
    wall=finish-begin;service=sum(phases.values());idle=wall-service
    if idle<-.01:raise ValueError('Stage accounting exceeds replay wall time')
    arrivals={e['id']:e for e in events if e['kind']=='arrival' and begin<=e['mono']<finish}
    identities={}
    for e in http:
        if e['kind']=='request_identity':
            for rid in e['ids']:identities[rid]=e['label']
    per_request=[];chunks=[]
    for rid,arrival in arrivals.items():
        own=[row for row in replay if any(r['id']==rid for r in row['requests'])]
        if not own:continue
        pf=[row for row in own if any(r['id']==rid and r['phase']=='prefill' for r in row['requests'])]
        ends=[(row,req) for row in own for req in row['completion_requests'] if req['id']==rid and req['finished']]
        if not ends:continue
        last,final=ends[-1]
        arrival_mono=arrival['mono']-(arrival['wall']-arrival['arrival'])
        first=own[0];prefill_end=pf[-1]['end'] if pf else first['start']
        first_meta=next(r for r in first['requests'] if r['id']==rid)
        label=identities.get(rid)
        join_method='response_id' if label else None
        if label is None:
            candidates=[e for e in http if e['kind']=='request_start' and e['label'].startswith('replay-') and e['prompt_tokens']==arrival['prompt'] and abs(e['wall']-arrival['arrival'])<10]
            if len(candidates)==1:
                label=candidates[0]['label'];join_method='unique_prompt_length_and_arrival_within_10s'
        entry={'label':label or 'unmatched-'+rid[:8],'join_method':join_method,'input_tokens':arrival['prompt'],'output_tokens':final['output'],
               'queue_s':max(0.,first['schedule_start']-arrival_mono),
               'admission_to_execute_s':max(0.,first['dispatch_start']-first['schedule_start']),
               'prefill_span_s':max(0.,prefill_end-first['dispatch_start']),
               'decode_span_s':max(0.,last['end']-prefill_end),
               'total_server_s':last['end']-arrival_mono,'prefill_chunks':len(pf),
               'prefix_cached_tokens':max(0,first_meta['computed']),
               'arrival_s_from_replay':arrival_mono-begin,
               'first_schedule_s_from_replay':first['schedule_start']-begin,
               'prefill_first_dispatch_s_from_replay':first['dispatch_start']-begin,
               'prefill_last_completion_s_from_replay':prefill_end-begin,
               'decode_first_dispatch_s_from_replay':next((row['dispatch_start']-begin for row in own if any(req['id']==rid and req['phase']=='decode' for req in row['requests'])),None),
               'finished_s_from_replay':last['end']-begin}
        per_request.append(entry)
        for row in pf:
            req=next(r for r in row['requests'] if r['id']==rid)
            chunks.append({'label':entry['label'],'step':row['step'],'computed_before':req['computed'],
                           'prompt_tokens_scheduled':min(req['scheduled'],req['prompt']-req['computed']),
                           'start_s_from_replay':row['start']-begin,'end_s_from_replay':row['end']-begin,
                           'phase':row['phase'],'shared_step_duration_s':row['duration_s']})
    latencies=[]
    for (batch,bucket),rows in sorted(groups.items()):
        latencies.append({'decode_batch':batch,'max_context_bin':bucket,'steps':len(rows),
                          'step_ms':quantiles([r['duration_s']*1000 for r in rows]),
                          'mean_step_ms':statistics.mean(r['duration_s']*1000 for r in rows),
                          'graph_hit_step_fraction':sum(r['graph_hit'] for r in rows)/len(rows)})
    prefix_before=next(e['prefix'] for e in http if e['kind']=='replay_start')
    prefix_after=next(e['prefix'] for e in http if e['kind']=='replay_end')
    prefix_delta={k:prefix_after.get(k,0)-v for k,v in prefix_before.items()}
    finished=[e for e in http if e['kind']=='request_end' and e['label'].startswith('replay-')]
    unmatched_sequences=set(validation['unmatched_worker_sequences'])
    validation['unmatched_replay_worker_sequences']=[e['seq'] for e in events if e['kind']=='execute_start' and e.get('rank')==0 and e['seq'] in unmatched_sequences and begin<=e['mono']<finish]
    validation['unmatched_outside_replay_count']=len(unmatched_sequences)-len(validation['unmatched_replay_worker_sequences'])
    validation.update({'http_completed':len(finished),'server_completed':len(per_request),
                       'all_requests_identified':all(not r['label'].startswith('unmatched') for r in per_request),
                       'request_count_matches':len(finished)==len(per_request),
                       'http_server_token_counts_match':all(any(e['label']==r['label'] and e['usage']['prompt_tokens']==r['input_tokens'] and e['usage']['completion_tokens']==r['output_tokens'] for e in finished) for r in per_request),
                       'no_negative_computed':all(c>=0 for row in replay for c in row['contexts']),
                       'prompt_tokens_accounted':all(sum(c['prompt_tokens_scheduled'] for c in chunks if c['label']==r['label'])+r['prefix_cached_tokens']==r['input_tokens'] for r in per_request),
                       'prefix_zero_hits':prefix_delta.get('prefix_cache_hits_total')==0,
                       'wall_partition_residual_s':wall-service-idle})
    summary={'clock':'CPU observed dispatch -> output completion; pipeline overlaps clipped; no added device synchronization',
             'wall_seconds':wall,'requests':len(per_request),'input_tokens':sum(r['input_tokens'] for r in per_request),
             'output_tokens':sum(r['output_tokens'] for r in per_request),
             'phase_seconds':dict(phases),'phase_steps':dict(phase_steps),'dispatch_idle_and_unassigned_seconds':idle,
             'decode_only_fraction_of_wall':phases['decode']/wall,'mixed_fraction_of_wall':phases['mixed']/wall,
             'decode_inclusive_mixed_fraction_of_wall':(phases['decode']+phases['mixed'])/wall,
             'decode_batch_seconds':dict(batch_seconds),'decode_batch_steps':dict(batch_count),'latency_by_batch_context':latencies,
             'graph_hit_decode_step_fraction':sum(r['graph_hit'] for r in replay if r['phase']=='decode')/max(1,phase_steps['decode']),
             'graph_hit_decode_time_fraction':sum(r['duration_s'] for r in replay if r['phase']=='decode' and r['graph_hit'])/max(1e-9,phases['decode']),
             'prefix_counter_delta':prefix_delta,
             'queue_seconds_quantiles':quantiles([r['queue_s'] for r in per_request]),
             'scheduler_cpu_seconds':sum(r['scheduler_cpu_s'] for r in replay),
             'worker_execute_host_span_seconds_includes_waits':sum(r['execute_cpu_span_s'] for r in replay),
             'worker_sample_host_span_seconds_includes_waits':sum(r['sample_cpu_span_s'] for r in replay),
             'validation':validation}
    return summary,per_request,chunks,replay


def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [],lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observation-dir',type=Path,required=True);p.add_argument('--http-events',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    events=[json.loads(line) for path in args.observation_dir.glob('*.jsonl') for line in path.read_text().splitlines()]
    http=[json.loads(line) for line in args.http_events.read_text().splitlines()]
    summary,requests,chunks,steps=analyze(events,http);args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'replay-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(args.out/'replay-requests.csv',requests);write_csv(args.out/'prefill-chunks.csv',chunks)
    flat=[{k:v for k,v in row.items() if k not in ('requests','completion_requests')} for row in steps]
    write_csv(args.out/'decode-steps.csv',flat)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
