#!/usr/bin/env python3
"""Sanitize kernel OOM evidence locally; do not publish machine identity or paths."""
import json,re
from pathlib import Path
ROOT=Path('results/decode-attribution-20261104/v3')
text=(ROOT/'raw/previous-boot-system-journal.txt').read_text();kernel=text[text.index('kernel: Mem-Info:'):]
table=kernel[kernel.index('[  pid  ]'):kernel.index('oom-kill:')];rows=[]
for line in table.splitlines():
 m=re.search(r'\[\s*(\d+)\]\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(.+)',line)
 if m:
  v=m.groups();rows.append({'uid':int(v[1]),'rss_GiB':int(v[4])*4096/2**30,'name':v[11]})
uid_python=[r for r in rows if r['uid']==1002 and r['name']=='python3']
victims={}
for timestamp,pid,rss in re.findall(r'(2026-11-04T\S+) .*?Killed process (\d+) \(python3\).*?anon-rss:(\d+)kB',text):victims[pid]={'time_utc':timestamp,'anon_rss_GiB':int(rss)/1024**2}
report={'last_previous_boot_journal_entry_utc':'2026-11-04T10:53:02Z','export_started_utc':'2026-11-04T09:39:01Z',
 'top_level_parallel_profile_analyses':8,'separate_cpu_or_memory_cgroup_limit_in_that_export':False,
 'first_logged_global_oom_utc':'2026-11-04T10:33:51Z','first_snapshot':{'anon_memory_GiB':(11061476+19745688)/1024**2,'free_memory_MiB':49654*4/1024,'swap_total_GiB':8388604/1024**2,'swap_free_KiB':96,'same_uid_python_processes':len(uid_python),'same_uid_python_rss_arithmetic_sum_GiB':sum(r['rss_GiB'] for r in uid_python),'rss_sum_warning':'Shared/forked pages counted repeatedly; this is not physical or peak memory; not every same-UID process can be assigned to the exporter from this snapshot'},
 'unique_python_oom_victims':len(victims),'python_oom_victim_anon_rss_GiB_range':[min(r['anon_rss_GiB'] for r in victims.values()),max(r['anon_rss_GiB'] for r in victims.values())],
 'other_confirmed_victims':['dbus-daemon','catatonit'],'network_daemon_invoked_oom_killer_utc':'2026-11-04T10:41:49Z',
 'evidence':'Multiple global_oom records; Python victims share the same rootless container cgroup; 8 top-level profiler analyses started together; 8 GiB swap exhausted.',
 'inference':'Unbounded parallel CPU export and its child tasks most likely exhausted host RAM/swap, causing severe memory pressure and loss of responsiveness.',
 'causal_limits':'First SSH/network failure was observed before the first recorded OOM. No continuous RSS/PSI/network samples bridge that interval; exact onset, aggregate exporter PSS peak and initial network failure mechanism cannot be proved. Journal alone does not establish a driver fault, kernel panic, or an automatic reboot.',
 'rule':'Large offline parsing stays on the local workstation with global memory limits and serial or <=2 parsing workers; experiment host collects and transfers only.'}
(ROOT/'incident-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
