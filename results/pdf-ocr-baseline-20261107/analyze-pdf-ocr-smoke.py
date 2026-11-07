"""Auxiliary PDF text-layer comparison, never invoice quality acceptance."""
import argparse, collections, hashlib, json, re, unicodedata
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--responses',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def normalize(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def distance(a,b):
 previous=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  current=[i]
  for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
  previous=current
 return previous[-1]
m=json.loads((a.inputs/'manifest.json').read_text());rows=[]
for c in m['cases']:
 response=a.responses/(c['id']+'-response.json')
 if not response.exists():continue
 v=json.loads(response.read_text());reference=(a.inputs/c['reference']).read_text();r=normalize(reference);h=normalize(v['text'])
 rd=collections.Counter(re.findall(r'\d+',r));hd=collections.Counter(re.findall(r'\d+',h));matched=sum((rd&hd).values())
 rows.append({'id':c['id'],'reference_chars':len(r),'recognized_chars':len(h),'normalized_edit_distance':distance(r,h),'auxiliary_cer':distance(r,h)/max(1,len(r)),'digit_runs_reference':sum(rd.values()),'digit_runs_recognized':sum(hd.values()),'digit_runs_overlap':matched,'finish_reason':v['finish_reason'],'truncated':v['finish_reason']=='length','response_sha256':hashlib.sha256(response.read_bytes()).hexdigest()})
out={'scope':'auxiliary OCR smoke metrics only; PDF embedded text may have reading-order and formula encoding errors; no human field ground truth or quality pass threshold','manual_ground_truth':False,'completed':len(rows),'cases':rows}
with a.out.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2)
