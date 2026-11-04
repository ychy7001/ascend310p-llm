#!/usr/bin/env python3
"""Streaming Chrome/Ascend trace primitives for bounded CPU attribution."""
import json
import re


def trace_events(stream, chunk_size=4*1024*1024):
    decoder=json.JSONDecoder();buffer='';position=0;eof=False;started=False
    while True:
        if not started:
            more=stream.read(chunk_size)
            if not more:
                raise ValueError('Trace event array not found')
            buffer+=more
            stripped=buffer.lstrip()
            if stripped.startswith('['):position=buffer.index('[')+1;started=True
            else:
                match=re.search(r'"traceEvents"\s*:\s*\[',buffer)
                if match:position=match.end();started=True
            if not started:continue
        while position<len(buffer) and buffer[position] in ' \r\n\t,':position+=1
        if position<len(buffer) and buffer[position]==']':return
        try:
            event,end=decoder.raw_decode(buffer,position)
        except json.JSONDecodeError:
            if eof:raise
            more=stream.read(chunk_size);buffer=buffer[position:]+more;position=0;eof=not bool(more)
            continue
        position=end
        if isinstance(event,dict):yield event
        if position>chunk_size:
            buffer=buffer[position:];position=0


def relative_timestamp(value,origin):
    text=str(value)
    if 'e' in text.lower():return float(text)-origin
    whole,dot,fraction=text.partition('.')
    return int(whole)-origin+(float('0.'+fraction) if dot else 0.)


def merge(intervals):
    result=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result


def union_duration(intervals):return sum(b-a for a,b in merge(intervals))


def intersection_duration(a,b):
    a=merge(a);b=merge(b);i=j=0;total=0.
    while i<len(a) and j<len(b):
        total+=max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total


def subtract(a,b):
    a=merge(a);b=merge(b);result=[];j=0
    for lo,hi in a:
        start=lo
        while j<len(b) and b[j][1]<=lo:j+=1
        k=j
        while k<len(b) and b[k][0]<hi:
            left,right=b[k]
            if left>start:result.append((start,min(left,hi)))
            start=max(start,right)
            if start>=hi:break
            k+=1
        if start<hi:result.append((start,hi))
    return result


def kernel_category(row):
    name=row.get('Name',row.get('Op Name',''))
    if 'hcom_allReduce' in name:return 'visible_allreduce'
    if 'hcom_allGather' in name:return 'visible_allgather'
    if 'QuantGroupedMatmulDequant' in name:return 'moe_grouped_w8a8'
    if name=='MatMul4/v2':return 'lm_head_fp16'
    if name.startswith('MatMulV2_NZ_NZ_FP16'):return 'fp16_nz_projection'
    if 'paged_attention_decoder_mask' in name:return 'paged_attention_decoder'
    if 'RecurrentGatedDeltaRule' in name:return 'recurrent_gdn'
    if 'TransData' in name:return 'layout_transdata'
    if 'QuantBatchMatmul' in name:return 'dense_quant_matmul'
    if any(x in name for x in ['MoeInitRouting','MoeGatingTopKSoftmax','MoeTokenUnpermute']):return 'moe_routing'
    return 'other_device_ops'
