#!/usr/bin/env python3
"""Allowlist-only inference log metadata extractor. Never returns log text or prompts."""
import ast
import datetime as dt
import hashlib
import json
import re
import sys

TIME = re.compile(r'^(\d{4}-\d\d-\d\dT\S+)')
NUMERIC = ('prompt_tokens', 'generated_tokens', 'completion_tokens', 'cached_tokens',
           'prefix_cache_hits', 'prefix_cache_queries', 'duration_ms', 'elapsed_s')


def parse_line(line):
    match = TIME.match(line)
    timestamp = match.group(1) if match else None
    result = []
    numbers = {}
    for name in NUMERIC:
        match = re.search(r'\b' + name + r'\s*[=:]\s*(\d+(?:\.\d+)?)\b', line)
        if match:
            numbers[name] = float(match.group(1))
    if 'prompt_tokens' in numbers and any(k in numbers for k in ('generated_tokens','completion_tokens')):
        # Completion summaries are distinguished from potential prompt text.
        if re.search(r'\bRequest finished: req \S+ finish_reason=', line) or re.search(r'\bRequest\s+\S+\s+(?:finished|completed)\b', line):
            request_match = re.search(r'\bRequest finished: req (\S+)', line) or re.search(r'\bRequest\s+(\S+)',line)
            request = request_match.group(1)
            result.append({'kind':'request_completed','time':timestamp,
                           'id':hashlib.sha256(request.encode()).hexdigest()[:20], **numbers})
    running = re.search(r'\bRunning:\s*(\d+)\s+reqs',line)
    if running and 'Avg prompt throughput' in line:
        waiting = re.search(r'\bWaiting:\s*(\d+)\s+reqs',line)
        prefix = re.search(r'\bPrefix cache hit rate:\s*(\d+(?:\.\d+)?)%',line)
        result.append({'kind':'concurrency_sample','time':timestamp,'running':int(running.group(1)),
                       'waiting':int(waiting.group(1)) if waiting else None,
                       'prefix_hit_percent':float(prefix.group(1)) if prefix else None})
    # Some gateways log a Python-literal transport summary with duration only.
    start = line.find('{')
    if start >= 0 and ('duration_ms' in line) and ('schema_version' in line):
        try:
            obj = ast.literal_eval(line[start:])
        except (SyntaxError,ValueError):
            obj = None
        if isinstance(obj,dict) and isinstance(obj.get('duration_ms'),(int,float)):
            result.append({'kind':'transport_completed','time':timestamp,'duration_ms':obj['duration_ms']})
    return result


def percentile(values, q):
    values=sorted(values)
    if not values:return None
    index=(len(values)-1)*q
    lo=int(index);hi=min(lo+1,len(values)-1)
    return values[lo]*(hi-index)+values[hi]*(index-lo) if hi!=lo else values[lo]


def quantiles(values):
    return {name:percentile(values,q) for name,q in [('p10',.1),('p25',.25),('p50',.5),('p75',.75),('p90',.9),('p95',.95),('p99',.99),('max',1)]}


def timestamp_seconds(text):
    if not text:return None
    return dt.datetime.fromisoformat(text.replace('Z','+00:00')).timestamp()


def sampled_histogram(rows, max_gap=20):
    """LOCF between adjacent samples only if gap <= max_gap; never invent idle time."""
    samples=sorted((timestamp_seconds(r['time']),int(r['running'])) for r in rows if r.get('time'))
    bins={};covered=0.;uncovered=0.
    for (a,n),(b,_) in zip(samples,samples[1:]):
        delta=b-a
        if delta<=0:continue
        if delta>max_gap:
            uncovered+=delta;continue
        bins[n]=bins.get(n,0.)+delta;covered+=delta
    return {'seconds':bins,'covered_seconds':covered,'unobserved_seconds':uncovered,
            'span_seconds':samples[-1][0]-samples[0][0] if len(samples)>1 else 0.,
            'method':'previous sample held only across gaps <=20 seconds; no interpolation across missing intervals'}


if __name__=='__main__':
    for line in sys.stdin:
        for row in parse_line(line):print(json.dumps(row,separators=(',',':')))


def interval_histogram(intervals, start=None, end=None):
    """Exact sweep of observed half-open intervals; caller supplies coverage limits."""
    intervals=[(float(a),float(b)) for a,b in intervals if b>a]
    if not intervals:return {'seconds':{},'span_seconds':0.}
    start=min(a for a,b in intervals) if start is None else start
    end=max(b for a,b in intervals) if end is None else end
    events={start:0,end:0}
    for a,b in intervals:
        a=max(a,start);b=min(b,end)
        if b<=a:continue
        events[a]=events.get(a,0)+1;events[b]=events.get(b,0)-1
    running=0;previous=start;bins={}
    for timestamp,delta in sorted(events.items()):
        if timestamp>previous:bins[running]=bins.get(running,0.)+timestamp-previous
        running+=delta;previous=timestamp
    return {'seconds':bins,'span_seconds':end-start}
