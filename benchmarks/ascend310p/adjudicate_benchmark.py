"""读取完整基准证据，按三臂几何中点和单侧 t 界判定。"""
import json
import math
from pathlib import Path
import statistics


def read_cell(path, prompt_tokens, concurrency, output_tokens=256, repeats=3):
    """校验单格 JSONL，返回请求时延和轮次吞吐的中位数。"""
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    config = records[0] if records else {}
    if config.get('kind') != 'config':
        raise ValueError(f'Expected first record kind=config, observed {config!r}')
    if (config.get('prompt_tokens'), config.get('concurrency')) != (prompt_tokens, concurrency):
        raise ValueError(f'Expected prompt_tokens/concurrency={prompt_tokens}/{concurrency}, '
                         f'observed {config!r}')
    prompt_ids = config.get('prompt_ids')
    if prompt_ids is None or len(prompt_ids) != prompt_tokens:
        raise ValueError(f'Expected {prompt_tokens} prompt_ids, observed {prompt_ids!r}')
    requests = [row for row in records if row.get('kind') == 'request']
    rounds = [row for row in records if row.get('kind') == 'round']
    if len(requests) != repeats * concurrency or len(rounds) != repeats:
        raise ValueError(f'Expected requests/rounds={repeats * concurrency}/{repeats}, '
                         f'observed {len(requests)}/{len(rounds)}')
    for index, row in enumerate(requests):
        if row.get('success') is not True:
            raise ValueError(f'Request {index}: expected success=True, observed {row!r}')
        usage = row.get('usage')
        if (usage is None or usage.get('prompt_tokens') != prompt_tokens
                or usage.get('completion_tokens') != output_tokens):
            raise ValueError(f'Request {index}: expected prompt/output={prompt_tokens}/{output_tokens}, '
                             f'observed usage={usage!r}')
    result = {'prompt_ids': prompt_ids}
    for metric, rows in (('ttft_s', requests), ('mean_tpot_s', requests),
                         ('aggregate_output_tok_s', rounds)):
        values = [row.get(metric) for row in rows]
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f'Expected finite positive {metric}, observed {values!r}')
        result[metric] = float(statistics.median(values))
    return result


def adjudicate(cells, rule):
    """以每格三臂对数比值的均值和给定 t 值判定三项指标。"""
    n = len(cells)
    if n < 2:
        raise ValueError(f'Expected at least 2 cells, observed {n}')
    expected = {'latency_delta', 'throughput_delta', 't_critical'}
    if set(rule) != expected:
        raise ValueError(f'Expected rule keys {sorted(expected)}, observed {sorted(rule)}')
    for key in ('latency_delta', 'throughput_delta'):
        value = rule[key]
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f'Expected finite {key} in (0, 1), observed {value!r}')
    t = rule['t_critical']['value']
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f'Expected finite positive t value, observed {t!r}')
    df = rule['t_critical']['degrees_of_freedom']
    if df != n - 1:
        raise ValueError(f'Expected degrees_of_freedom={n - 1}, observed {df!r}')
    seen = set()
    for cell in cells:
        key = (cell['prompt_tokens'], cell['concurrency'])
        if key in seen:
            raise ValueError(f'Duplicate cell {key!r}')
        seen.add(key)
        for arm in ('A', 'B', 'A_prime'):
            for metric in ('ttft_s', 'mean_tpot_s', 'aggregate_output_tok_s'):
                value = cell[arm][metric]
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f'Cell {key!r} {arm}: expected finite positive {metric}, '
                                     f'observed {value!r}')
        ids = {arm: cell[arm]['prompt_ids'] for arm in ('A', 'B', 'A_prime')}
        if not ids['A'] == ids['B'] == ids['A_prime']:
            raise ValueError(f'Cell {key!r}: different prompt_ids {ids!r}')
    metrics = {}
    cell_results = [{'prompt_tokens': cell['prompt_tokens'], 'concurrency': cell['concurrency'],
                     'log_ratio': {}} for cell in cells]
    for metric in ('ttft_s', 'mean_tpot_s', 'aggregate_output_tok_s'):
        logs = [math.log(cell['B'][metric] / math.sqrt(cell['A'][metric] * cell['A_prime'][metric]))
                for cell in cells]
        for cell_result, value in zip(cell_results, logs):
            cell_result['log_ratio'][metric] = value
        mean = statistics.mean(logs)
        sd = statistics.stdev(logs)
        se = sd / math.sqrt(n)
        if metric == 'aggregate_output_tok_s':
            bound = math.exp(mean - t * se)
            threshold = 1 - rule['throughput_delta']
            passed = bound >= threshold
        else:
            bound = math.exp(mean + t * se)
            threshold = 1 + rule['latency_delta']
            passed = bound <= threshold
        metrics[metric] = {'pooled_ratio': math.exp(mean), 'log_sd': sd,
                           'standard_error': se, 'bound': bound,
                           'threshold': threshold, 'passed': passed}
    return {'passed': all(row['passed'] for row in metrics.values()),
            'cells': cell_results, 'metrics': metrics,
            'reference': 'B / sqrt(A * A_prime) pooled over cells; one-sided t bound'}
