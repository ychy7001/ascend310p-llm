#!/usr/bin/env python3
"""Original BF16 Qwen3.5 MoE checkpoint reference, layer-at-a-time FP32 CPU.

Run in the pinned Transformers CPU environment. No model downloads, quantized
operators, CUDA or NPU. All prompt tokens are retained. Attention is evaluated
with bounded query blocks; GDN keeps the upstream 64-token chunk arithmetic.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
import transformers
import transformers.utils as transformer_utils
import transformers.utils.import_utils as transformer_imports

# This CPU image also distributes torch_npu; discovery would load a driver.
transformer_imports.is_torch_npu_available = lambda *args, **kwargs: False
transformer_utils.is_torch_npu_available = transformer_imports.is_torch_npu_available
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

UPSTREAM_ATTENTION = m.eager_attention_forward
UPSTREAM_GDN = m.torch_chunk_gated_delta_rule


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def bounded_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    if dropout != 0 or attention_mask is not None:
        raise ValueError('Reference accepts only unpadded evaluation with implicit causal mask')
    key = m.repeat_kv(key, module.num_key_value_groups)
    value = m.repeat_kv(value, module.num_key_value_groups)
    length = query.shape[-2]
    if key.shape[-2] != length:
        raise ValueError('Reference is full-context prefill, without a KV cache')
    result = torch.empty_like(query)
    for start in range(0, length, 128):
        stop = min(start + 128, length)
        scores = torch.matmul(query[:, :, start:stop], key[:, :, :stop].transpose(2, 3)) * scaling
        future = torch.arange(stop)[None, :] > torch.arange(start, stop)[:, None]
        scores.masked_fill_(future, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        result[:, :, start:stop] = torch.matmul(probs, value[:, :, :stop])
        del scores, probs
    return result.transpose(1, 2).contiguous(), None


def bounded_gdn(query, key, value, g, beta, chunk_size=64, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=False):
    if chunk_size != 64:
        raise ValueError('Frozen upstream GDN chunk size is 64')
    result = torch.empty_like(value)
    state = initial_state
    for start in range(0, query.shape[1], 512):
        stop = min(start + 512, query.shape[1])
        out, state = UPSTREAM_GDN(query[:, start:stop], key[:, start:stop], value[:, start:stop],
                                 g[:, start:stop], beta[:, start:stop], chunk_size=64,
                                 initial_state=state, output_final_state=True,
                                 use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        result[:, start:stop] = out
    return result, state if output_final_state else None


def setup(threads):
    if 'torch_npu' in sys.modules:
        raise RuntimeError('CPU reference unexpectedly imported torch_npu')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    for name in ('chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule',
                 'causal_conv1d_fn', 'causal_conv1d_update'):
        setattr(m, name, None)
    m.eager_attention_forward = bounded_attention


class Weights:
    def __init__(self, directory):
        self.directory = directory
        self.index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
        self.config = Qwen3_5MoeTextConfig(**json.loads((directory / 'config.json').read_text())['text_config'])
        self.config._attn_implementation = 'eager'
        self.config._experts_implementation = 'eager'
        for shard in set(self.index.values()):
            if not (directory / shard).is_file():
                raise ValueError('Incomplete checkpoint: ' + shard)

    def tensor(self, name):
        with safe_open(self.directory / self.index[name], framework='pt', device='cpu') as source:
            return source.get_tensor(name)

    def load(self, module, prefix):
        for name, parameter in list(module.named_parameters()):
            raw = self.tensor(prefix + name)
            if raw.dtype != torch.bfloat16 or raw.shape != parameter.shape:
                raise ValueError('Checkpoint tensor mismatch: ' + prefix + name)
            parent, _, leaf = name.rpartition('.')
            owner = module.get_submodule(parent) if parent else module
            setattr(owner, leaf, torch.nn.Parameter(raw.float(), requires_grad=False))
        return module.eval()

    def layer(self, index):
        with torch.device('meta'):
            layer = m.Qwen3_5MoeDecoderLayer(self.config, index)
        self.load(layer, f'model.language_model.layers.{index}.')
        if layer.layer_type == 'linear_attention':
            layer.linear_attn.chunk_gated_delta_rule = bounded_gdn
            layer.linear_attn.recurrent_gated_delta_rule = m.torch_recurrent_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is None
        return layer

    def embed(self, ids):
        return F.embedding(torch.tensor([ids]), self.tensor('model.language_model.embed_tokens.weight')).float()

    def logits(self, hidden):
        key = 'lm_head.weight'
        with safe_open(self.directory / self.index[key], framework='pt', device='cpu') as source:
            weight = source.get_slice(key)
            size = weight.get_shape()[0]
            result = torch.empty(hidden.shape[0], size, dtype=torch.float32)
            for start in range(0, size, 8192):
                stop = min(size, start + 8192)
                result[:, start:stop] = F.linear(hidden, weight[start:stop].float())
        return result


def kernel_checks():
    torch.manual_seed(310)
    class Attention:
        num_key_value_groups = 2
        training = False
    q = torch.randn(1, 4, 273, 16)
    k, v = (torch.randn(1, 2, 273, 16) for _ in range(2))
    mask = torch.zeros(273, 273).masked_fill(torch.ones(273, 273, dtype=torch.bool).triu(1), torch.finfo(torch.float32).min)
    expected, _ = UPSTREAM_ATTENTION(Attention(), q, k, v, mask, scaling=.25)
    actual, _ = bounded_attention(Attention(), q, k, v, None, scaling=.25)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    attention_gap = float((actual - expected).abs().max())
    # Cross both the 64-token and 512-token boundaries, including a partial tail.
    q, k, v = (torch.randn(1, 577, 2, 8) for _ in range(3))
    g = -torch.rand(1, 577, 2) * .1
    beta = torch.rand(1, 577, 2)
    kwargs = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    expected, es = UPSTREAM_GDN(q, k, v, g, beta, **kwargs)
    actual, state = bounded_gdn(q, k, v, g, beta, **kwargs)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(state, es, atol=3e-6, rtol=3e-5)
    # Independent recurrent formulation checks the same non-padded sequence.
    recurrent, rs = m.torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=None, **kwargs)
    torch.testing.assert_close(actual, recurrent, atol=3e-6, rtol=3e-5)
    # Tail padding decays by exp(0) and beta=0, so the final state must match.
    torch.testing.assert_close(state, rs, atol=3e-6, rtol=3e-5)
    return {'passed': True, 'attention_max_abs': attention_gap,
            'gdn_upstream_max_abs': float((actual - expected).abs().max()),
            'gdn_recurrent_max_abs': float((actual - recurrent).abs().max())}


def run(args):
    args.out.mkdir(exist_ok=False)
    setup(args.threads)
    write(args.out / 'kernel-checks.json', kernel_checks())
    weights = Weights(args.model)
    inputs = json.loads(args.inputs.read_text())
    selection = json.loads(args.protocol.read_text())['cpu_subset_protocol']
    if sha(args.inputs) != selection['input_sha256']:
        raise ValueError('Frozen input hash changed')
    targets = {}
    if not args.diagnostic:
        ref = json.loads(args.continuation.read_text())
        frozen = json.loads(args.continuation_freeze.read_text())
        if sha(args.continuation) != frozen['sha256'] or frozen['R_read'] or frozen['B_read']:
            raise ValueError('A continuation was not frozen before R/B')
        targets = {row['id']: row['token_ids'] for row in ref['cases']}
        if len(targets) != 35 or any(len(ids) != 256 for ids in targets.values()):
            raise ValueError('Expected all 35 fresh 256-token A continuations')
    selected = {row['case']: row['stop_exclusive'] for row in selection['positions']}
    cases = [row for row in inputs['cases'] if not args.cases or row['id'] in args.cases]
    if not cases or (not args.diagnostic and len(cases) != 35):
        raise ValueError('Full R requires all frozen 35 cases')
    write(args.out / 'freeze.json', {'diagnostic_only': args.diagnostic, 'script_sha256': sha(Path(__file__)),
        'input_sha256': sha(args.inputs), 'protocol_sha256': sha(args.protocol),
        'continuation_sha256': sha(args.continuation) if not args.diagnostic else None,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'threads': args.threads,
        'weights_dtype': 'original_BF16', 'compute_dtype': 'FP32', 'device': 'cpu',
        'layers': weights.config.num_hidden_layers, 'case_ids': [c['id'] for c in cases], 'unix_ns': time.time_ns()})
    hidden = {}
    lengths = {}
    for case in cases:
        # N target logits need prompt plus N-1 continuation tokens, inclusive of
        # the final prompt position which predicts the first continuation token.
        ids = case['token_ids'] + (targets[case['id']][:selected[case['id']] - 1] if targets else [])
        hidden[case['id']] = weights.embed(ids)
        lengths[case['id']] = len(case['token_ids'])
    rotary = m.Qwen3_5MoeTextRotaryEmbedding(weights.config)
    start = time.monotonic()
    for index in range(weights.config.num_hidden_layers):
        layer = weights.layer(index)
        for case in cases:
            case_id = case['id']
            atomic(args.out / 'status.json', {'state': 'running', 'layer': index, 'case': case_id,
                'elapsed_s': time.monotonic() - start, 'unix_ns': time.time_ns(),
                'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            hs = hidden[case_id]
            positions = torch.arange(hs.shape[1]).view(1, -1)
            embeddings = rotary(hs, positions)
            then = time.monotonic()
            hidden[case_id] = layer(hs, position_embeddings=embeddings, attention_mask=None, past_key_values=None)
            if not torch.isfinite(hidden[case_id]).all():
                raise ValueError(f'Nonfinite hidden state: {index} {case_id}')
            print(json.dumps({'layer': index, 'case': case_id, 'tokens': hs.shape[1],
                'compute_s': time.monotonic() - then, 'elapsed_s': time.monotonic() - start,
                'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
            del hs, embeddings
        del layer
        gc.collect()
    with torch.device('meta'):
        norm = m.Qwen3_5MoeRMSNorm(weights.config.hidden_size, weights.config.rms_norm_eps)
    weights.load(norm, 'model.language_model.norm.')
    result = {'arm': 'R', 'diagnostic_only': args.diagnostic, 'mode': 'strict_teacher_forcing',
              'input_sha256': sha(args.inputs), 'continuation_sha256': sha(args.continuation) if targets else None, 'cases': []}
    for case in cases:
        case_id = case['id']; n = selected[case_id] if targets else 1
        hs = norm(hidden.pop(case_id)[:, lengths[case_id] - 1: lengths[case_id] - 1 + n]).squeeze(0)
        logprobs = F.log_softmax(weights.logits(hs), dim=-1)
        values, indices = torch.topk(logprobs, 20, dim=-1)
        positions = []
        for pos in range(n):
            context = case['token_ids'] + (targets[case_id][:pos] if targets else [])
            target = targets[case_id][pos] if targets else None
            positions.append({'position': pos, 'top1': int(indices[pos, 0]),
                'logprobs': {str(int(k)): float(v) for k, v in zip(indices[pos], values[pos])},
                'target': target, 'target_logprob': float(logprobs[pos, target]) if target is not None else None,
                'target_upper_bound': None,
                'context_sha256': hashlib.sha256(json.dumps(context, separators=(',', ':')).encode()).hexdigest()})
        result['cases'].append({'id': case_id, 'target_token_ids': targets[case_id][:n] if targets else [], 'positions': positions})
        atomic(args.out / 'R.partial.json', result)
    write(args.out / 'R.json', result)
    atomic(args.out / 'status.json', {'state': 'complete', 'diagnostic_only': args.diagnostic,
        'cases': len(cases), 'positions': sum(len(c['positions']) for c in result['cases']),
        'elapsed_s': time.monotonic() - start, 'unix_ns': time.time_ns(),
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'inputs', 'protocol', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--continuation', type=Path)
    p.add_argument('--continuation-freeze', type=Path)
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--cases', nargs='+')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if not args.diagnostic and (args.continuation is None or args.continuation_freeze is None):
        p.error('Full R requires fresh frozen A continuation and its freeze record')
    try:
        with torch.inference_mode():
            run(args)
    except BaseException as exc:
        if args.out.is_dir():
            atomic(args.out / 'status.json', {'state': 'failed', 'error': repr(exc), 'unix_ns': time.time_ns()})
        raise


if __name__ == '__main__':
    main()
