"""从配对实验核算上下文档位；不执行/修改 NPU 服务。"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'clt-kv-comparison-20261015'
account = json.loads((SOURCE / 'memory-accounting.json').read_text())
summary = json.loads((SOURCE / 'summary.json').read_text())
GIB = 1 << 30
# npu-smi 的 MB 数字在本预算中按 1024 换算为 GiB，原始 MB 同时保留。
total_mb = 44215
peaks_mb = {'c16': 40561, 'c8': 35518}
base_gib = max(
    peaks_mb['c16'] / 1024 - account['c16_auto_unique_storage_bytes'] / GIB,
    peaks_mb['c8'] / 1024 - (account['c8_auto_pool_bytes'] + account['c8_shared_scratch_bytes']) / GIB,
)
margin_gib = 2.0  # 工程建议值，尚未经压力/并发测试确定。
rows = []
for label, limit in [('8K', 8320), ('32K', 32896), ('64K', 65664), ('96K', 98432), ('112K', 114688)]:
    # 冻结已审计的 1152-token 页、三个 GDN 组各两个页以及 null 页。
    blocks = math.ceil(limit / 1152) + 3 * 2 + 1
    scratch = math.ceil(limit / 64) * 64 * 2048
    c16 = base_gib + blocks * account['c16_actual_bytes_per_scheduler_block'] / GIB + margin_gib
    c8 = base_gib + (blocks * account['c8_actual_bytes_per_scheduler_block'] + scratch) / GIB + margin_gib
    rows.append({'label': label, 'total_context_cap': limit, 'scheduler_blocks_assumed': blocks,
                 'c16_planning_reservation_gib': c16, 'c16_uncommitted_gib': total_mb / 1024 - c16,
                 'c8_planning_reservation_gib': c8, 'c8_uncommitted_gib': total_mb / 1024 - c8,
                 'c8_saving_gib': c16 - c8})
result = {'status': 'capacity estimates; no service run at these minimum block counts',
          'device_total_mb': total_mb, 'sampled_peaks_mb': peaks_mb,
          'memory_unit_convention': 'npu-smi MB / 1024 used for GiB planning; raw MB retained',
          'non_pool_baseline_gib': base_gib, 'assumed_safety_margin_gib': margin_gib,
          'profiles': rows}
(ROOT / 'capacity-estimates.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
lines = ['| 业务档 | 总上下文预算 | C16预留 / 余量 GiB | C8预留 / 余量 GiB | C8净省 GiB |',
         '|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['label']} | {r['total_context_cap']} | {r['c16_planning_reservation_gib']:.2f} / {r['c16_uncommitted_gib']:.2f} | {r['c8_planning_reservation_gib']:.2f} / {r['c8_uncommitted_gib']:.2f} | {r['c8_saving_gib']:.2f} |")
(ROOT / 'capacity-table.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
for row in summary['short']:
    print(row['prompt_tokens'], 'decode loss vs 8K:',
          {kind: 100 * (1-row[kind]['decode_tok_s']/summary['short'][0][kind]['decode_tok_s']) for kind in ('c16', 'c8')})
