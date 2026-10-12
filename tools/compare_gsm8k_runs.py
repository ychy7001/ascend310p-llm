# SPDX-License-Identifier: Apache-2.0
"""Compare only completed GSM HTTP runs with identical data and prompt protocol."""
import argparse
import json
from pathlib import Path


def read(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if rows[0]['kind']!='config' or rows[-1]['kind']!='summary':
        raise ValueError(f'Run incomplete: {path}')
    answers={r['index']:r for r in rows if r['kind']=='answer'}
    if len(answers)!=len(rows[0]['indices']) or len(answers)!=rows[-1]['samples']:
        raise ValueError(f'Sample count mismatch: {path}')
    return rows[0],answers,rows[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    a=p.parse_args();ref,ra,rs=read(a.reference);cand,ca,cs=read(a.candidate)
    for key in ['dataset_sha256','dataset_rows','indices','max_tokens','temperature',
                'enable_thinking','prompt_suffix','protocol']:
        if ref[key]!=cand[key]:raise ValueError(f'Incompatible {key}')
    changes=[]
    for index in ref['indices']:
        r,c=ra[index],ca[index]
        if r['gold']!=c['gold']:raise ValueError('Gold differs')
        if r['correct']!=c['correct'] or r['predicted']!=c['predicted']:
            changes.append({'index':index,'gold':r['gold'],'reference':r['predicted'],
                'candidate':c['predicted'],'reference_finish':r['finish_reason'],
                'candidate_finish':c['finish_reason']})
    print(json.dumps({'reference':rs,'candidate':cs,'answer_changes':changes,
        'identical_texts':sum(ra[i]['text']==ca[i]['text'] for i in ref['indices']),
        'limitation':'small fixed subset; not comprehensive quality or statistical equivalence'},indent=2))


if __name__=='__main__':main()
