#!/usr/bin/env python3
"""Export accounting CSV from existing CANN device timelines, on CPU only.

This does not decode raw binary captures or synthesize timings. Shape fields absent
from the timeline remain N/A. Eager MatMul4/v2 attribution requires native
op_summary tensor shapes; this narrow export cannot supply that distinction. It is a narrow kernel accounting export, not the full
PyTorch profiler report. Input files must contain disjoint event sources.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from decode_trace_utils import trace_events

FIELDS=['Name','Start Time(us)','Duration(us)','Model ID','Stream ID','Task ID','Input Shapes','Input Data Types','Input Formats']
COMPUTE_TYPES={'AI_CORE','AI_CPU','AI_VECTOR','MIX_AIC','MIX_AIV','MIX_AI_CORE'}

def kernel_row(event):
    if event.get('ph')!='X':return None
    args=event.get('args',{});name=event.get('name','')
    if name in COMPUTE_TYPES:return None  # Unnamed hardware task is absent from op_summary.
    if args.get('Task Type') not in COMPUTE_TYPES and not name.startswith(('hcom_allReduce','hcom_allGather')):return None
    if float(event.get('dur',0))<0:raise ValueError('Negative kernel duration')
    return {'Name':name,'Start Time(us)':str(event['ts']),'Duration(us)':str(event['dur']),
            'Model ID':args.get('Model Id',args.get('model id','N/A')),
            'Stream ID':args.get('Physic Stream Id',event.get('tid','N/A')),
            'Task ID':args.get('Task Id','N/A'),'Input Shapes':'N/A','Input Data Types':'N/A','Input Formats':'N/A'}

def export(sources,target):
    counts=Counter();events=0
    with target.open('w') as output:
        writer=csv.DictWriter(output,fieldnames=FIELDS,lineterminator='\n');writer.writeheader()
        for source in sources:
            with source.open() as stream:
                for event in trace_events(stream):
                    events+=1;row=kernel_row(event)
                    if row is not None:writer.writerow(row);counts[event.get('args',{}).get('Task Type','collective_wrapper')]+=1
    return {'events_scanned':events,'kernel_rows':sum(counts.values()),'kernel_types':dict(counts),
            'source':'Existing CANN timeline X events; timestamps/durations unchanged; no raw binary reparse',
            'limitations':'No absent tensor shapes inferred; no host/framework trace reconstructed'}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--timeline',type=Path,nargs='+',required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--metadata',type=Path,required=True);args=parser.parse_args()
    report=export(args.timeline,args.out);args.metadata.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
