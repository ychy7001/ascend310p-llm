#!/usr/bin/env python3
"""Streaming Chrome/Ascend trace primitives for bounded CPU attribution."""
import json
import re


def trace_events(stream, chunk_size=4*1024*1024, allow_missing_final_bracket=False):
    decoder=json.JSONDecoder();buffer='';position=0;eof=False;started=False;top_level_array=False;event_count=0;comma_after_event=False
    while True:
        if not started:
            more=stream.read(chunk_size)
            if not more:
                raise ValueError('Trace event array not found')
            buffer+=more
            stripped=buffer.lstrip()
            if stripped.startswith('['):position=buffer.index('[')+1;started=True;top_level_array=True
            else:
                match=re.search(r'"traceEvents"\s*:\s*\[',buffer)
                if match:position=match.end();started=True
            if not started:continue
        while position<len(buffer) and buffer[position] in ' \r\n\t,':
            if buffer[position]==',':comma_after_event=True
            position+=1
        if position<len(buffer) and buffer[position]==']':return
        try:
            event,end=decoder.raw_decode(buffer,position)
        except json.JSONDecodeError:
            if eof:
                # Optional recovery of an event-complete, unclosed top-level array.
                # An incomplete event/object still raises; no event is synthesized.
                if allow_missing_final_bracket and top_level_array and event_count and not comma_after_event and not buffer[position:].strip():return
                raise
            more=stream.read(chunk_size);buffer=buffer[position:]+more;position=0;eof=not bool(more)
            continue
        position=end;event_count+=1;comma_after_event=False
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
    if name=='MatMul4/v2':
        # Eager uses the same name for all 220 projections and the vocabulary head.
        # CANN reports physical NZ tensors as [K/16,N/16,16,16]. Graph head
        # rows may have N/A shape; graph projections have distinct kernel names.
        shape=row.get('Input Shapes','').strip().strip(chr(34))
        if shape and shape not in ('N/A','NA'):
            tensors=[part.strip().strip(chr(34)) for part in shape.split(';') if part.strip()]
            try:
                weight=[int(v) for v in tensors[-1].split(',')]
            except (ValueError,IndexError):
                raise ValueError('Unrecognized MatMul4/v2 shape: '+shape)
            if len(weight)==4 and weight[-2:]==[16,16]:
                return 'lm_head_fp16' if weight[1]*16>65536 else 'fp16_nz_projection'
            if len(weight)==2:
                return 'lm_head_fp16' if max(weight)>65536 else 'fp16_nz_projection'
            raise ValueError('Unrecognized MatMul4/v2 weight shape: '+shape)
        return 'lm_head_fp16'
    if name.startswith('MatMulV2_NZ_NZ_FP16'):return 'fp16_nz_projection'
    if 'paged_attention_decoder_mask' in name:return 'paged_attention_decoder'
    if 'RecurrentGatedDeltaRule' in name:return 'recurrent_gdn'
    if 'TransData' in name:return 'layout_transdata'
    if 'QuantBatchMatmul' in name:return 'dense_quant_matmul'
    if any(x in name for x in ['MoeInitRouting','MoeGatingTopKSoftmax','MoeTokenUnpermute']):return 'moe_routing'
    return 'other_device_ops'
