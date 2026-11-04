#!/usr/bin/env python3
"""Transfer measured per-case fractions using decode time weights, never sum ranks."""
import argparse,json
from pathlib import Path

def weighted_profiles(profiles,time_weights,decode_wall_fraction,mean_replay_step_ms):
    if set(profiles)!=set(time_weights):raise ValueError('Case/weight mismatch')
    if any(w<0 for w in time_weights.values()) or sum(time_weights.values())<=0:raise ValueError('Invalid weights')
    total=sum(time_weights.values());weights={k:v/total for k,v in time_weights.items()}
    names=set().union(*(p['categories'] for ranks in profiles.values() for p in ranks))
    fractions={k:0. for k in sorted(names)};fractions.update(named_kernel_gap=0.,category_overlap_correction=0.)
    groups={};detail=[]
    for case,ranks in profiles.items():
        if len(ranks)!=2:raise ValueError('Exactly two ranks required')
        for rank,p in enumerate(ranks):
            cycle=p['window_ms_per_step']
            if cycle<=0:raise ValueError('Nonpositive cycle')
            factor=weights[case]/len(ranks)/cycle
            cat_total=0.
            for name,item in p['categories'].items():
                cat_total+=item['union_ms_per_step'];fractions[name]+=factor*item['union_ms_per_step']
            fractions['named_kernel_gap']+=factor*p['uncovered_by_named_kernel_ms_per_step']
            fractions['category_overlap_correction']+=factor*(p['named_kernel_union_ms_per_step']-cat_total)
            for name,value in p['fp16_module_attribution']['groups_ms_per_step'].items():groups[name]=groups.get(name,0.)+factor*value
            detail.append({'case':case,'rank':rank,'steps':p['steps'],'cycle_ms':cycle,'time_weight':weights[case]})
    def present(fraction):
        return {'decode_time_fraction':fraction,'equivalent_ms_per_replay_step':fraction*mean_replay_step_ms,
                'replay_wall_fraction':fraction*decode_wall_fraction}
    return {'method':'Mean of per-rank per-case fractions, then weighted by decode wall time; ranks are not summed',
            'time_weights':weights,'decode_wall_fraction':decode_wall_fraction,'mean_replay_step_ms':mean_replay_step_ms,
            'cases':detail,'composition':{k:present(v) for k,v in fractions.items()},
            'fp16_groups':{k:present(v) for k,v in groups.items()},'fraction_sum':sum(fractions.values())}

def scenario(component,speedup):
    if speedup<1:raise ValueError('Speedup must be >=1')
    pool=component['replay_wall_fraction']
    return {'removable_kernel_wall_fraction':pool,'speedup':speedup,'wall_reduction_fraction':pool*(1-1/speedup),
            'minimum_kernel_speedup_for_3pct_wall_ignoring_overheads':1/(1-.03/pool) if pool>.03 else None}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profiles',type=Path,required=True);parser.add_argument('--replay',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    replay=json.loads(a.replay.read_text());labels=['b1-long','b2-long','b3-short','mixed-b4']
    profiles={label:[json.loads((a.profiles/f'{label}-rank{rank}.json').read_text()) for rank in (0,1)] for label in labels}
    weights={label:replay['decode_batch_seconds'][str(i+1)] for i,label in enumerate(labels)}
    result=weighted_profiles(profiles,weights,replay['decode_only_fraction_of_wall'],replay['phase_seconds']['decode']*1000/replay['phase_steps']['decode'])
    result['candidate_scenarios']={name:scenario(component,2.) for name,component in {**result['composition'],**result['fp16_groups']}.items() if component['replay_wall_fraction']>0}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
