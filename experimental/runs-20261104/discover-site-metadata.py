#!/usr/bin/env python3
"""Remote schema-only inspection; never emits log text or request contents."""
import concurrent.futures,subprocess,pathlib
root=pathlib.Path('results/decode-attribution-20261104/v2/raw')
script=r'''import subprocess,json
pfx=['sudo','-n'] if SITE=='root' else []
p=subprocess.run(pfx+['podman','ps','--format','{{.Names}}'],capture_output=True,text=True)
for n in [n for n in p.stdout.splitlines() if 'llm' in n]:
 r=subprocess.run(pfx+['podman','logs','--tail','2000',n],capture_output=True,text=True)
 lines=(r.stdout+r.stderr).splitlines(); schemas={}; counts={}; patterns=[]
 for line in lines:
  try:
   d=json.loads(line); key=','.join(sorted(d)); schemas[key]=schemas.get(key,0)+1
   for k,v in d.items():
    if isinstance(v,(int,float)):counts[k]=type(v).__name__
  except Exception:
   for pat in ['prompt_tokens','completion_tokens','input_tokens','output_tokens','prefix','cached','request','Request','Running','Avg prompt','POST','usage']:
    if pat in line: patterns.append(pat)
 print(json.dumps({'kind':'gateway' if 'gateway' in n else 'backend','lines':len(lines),'schemas':schemas,'numeric_keys':counts,'patterns':{k:patterns.count(k) for k in set(patterns)}}))
'''
def run(host,kind):
 p=subprocess.run(['ssh',host,'python3 -'],input='SITE='+repr(kind)+'\n'+script,text=True,capture_output=True,timeout=90)
 (root/(host+'-schema.txt')).write_text(p.stdout+p.stderr)
 print(host,p.returncode,p.stdout[:10000],p.stderr[:500],flush=True)
with concurrent.futures.ThreadPoolExecutor() as ex:list(ex.map(lambda x:run(*x),[('ychy-a1','root'),('ails-n2','user'),('ails-n3','user')]))
