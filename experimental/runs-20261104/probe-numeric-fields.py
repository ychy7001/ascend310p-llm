import subprocess,concurrent.futures,json,pathlib
script=r'''import subprocess,re,json,hashlib
pfx=['sudo','-n'] if ROOT else []
names=subprocess.check_output(pfx+['podman','ps','--format','{{.Names}}'],text=True).splitlines()
for n in names:
 if n!='llm-backend-primary' and n!='llm-gateway':continue
 p=subprocess.run(pfx+['podman','logs','--timestamps',n],capture_output=True,text=True)
 out=[]; field_counts={}; timestamps=[]; examples=[]
 for line in (p.stdout+p.stderr).splitlines():
  ts=re.match(r'\d{4}-\d\d-\d\dT\S+',line)
  if ts:timestamps.append(ts.group())
  fields=re.findall(r'([A-Za-z_][A-Za-z_ ]{0,35})\s*[=:]\s*(-?\d+(?:\.\d+)?)',line)
  safe=[(k.strip(),float(v)) for k,v in fields if any(w in k.lower() for w in ['token','running','waiting','cache','time','latency','request','ms','prefix','batch','queue'])]
  perc=re.findall(r'([A-Za-z_][A-Za-z_ ]{0,35})\s*:\s*(\d+(?:\.\d+)?)%',line)
  safe+=perc
  if safe:
   out.append({'time':ts.group() if ts else None,'fields':dict(safe),'completion':bool(re.search(r'Request.*(finished|completed|prompt_tokens)',line))})
   for k,v in safe:field_counts[k]=field_counts.get(k,0)+1
 print(json.dumps({'kind':'gateway' if 'gateway' in n else 'backend','line_count':len((p.stdout+p.stderr).splitlines()),'first':min(timestamps) if timestamps else None,'last':max(timestamps) if timestamps else None,'field_counts':field_counts,'examples':out[-5:]}))
 open(OUT+'-'+('gateway' if 'gateway' in n else 'backend')+'.json','w').write(json.dumps(out)) if False else None
 print('NUMERIC_RECORDS '+json.dumps(out))
'''
root=pathlib.Path('results/decode-attribution-20261104/v2/raw')
def run(h,r):
 p=subprocess.run(['ssh',h,'python3 -'],input='ROOT='+repr(r)+'\n'+script,capture_output=True,text=True,timeout=120)
 (root/(h+'-numeric.jsonl')).write_text(p.stdout)
 for l in p.stdout.splitlines():
  if not l.startswith('NUMERIC_RECORDS'): print(h,l,flush=True)
 print(h,'exit',p.returncode,p.stderr[:300],flush=True)
with concurrent.futures.ThreadPoolExecutor() as ex:list(ex.map(lambda x:run(*x),[('ychy-a1',True),('ails-n2',False),('ails-n3',False)]))
