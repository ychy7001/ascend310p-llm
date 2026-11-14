# [310P] compressed SplitFuse v2 changes causal output when only future V is modified

Status: prepared report; not submitted upstream.

## Problem

On a 310P3, `_npu_paged_attention_splitfuse_v2` with `mask_type=5` and the fixed ND `[2048,2048]` upper-triangular mask can expose future V for a non-block-aligned context length. Legacy `_npu_paged_attention_splitfuse` with an explicit causal mask preserves causality under identical Q/K/V and block tables.

This is a direct torch-npu operator repro. No vLLM import, model weights, GDN, WY custom operation, speculative decoding, or serving scheduler is required.

## Environment tested

- Atlas 300I DUO, two 310P3 chips; reproduced independently on devices 0 and 1.
- CANN 9.2.1; PyTorch `2.10.0+cpu`; torch-npu `2.10.0.post7`.
- Reproduced in a retained vllm-ascend main snapshot with image revision `76ea9d24f155afe032ad9ac6c98717f1f7334f36` and vLLM source revision `4ba787ec9959e566d3b587a796dff07673324da5`.
- Also reproduced in our older v0.30 integration using the same CANN/torch-npu stack.
- These are pinned locally available snapshots, not a claim that the current upstream head has been tested.

## Reproduction

Run `repro_future_mask.py` from this directory in the initialized Ascend environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 repro_future_mask.py --device 0
```

The script is standalone and includes both the legacy control and the compressed operation. Query length=384, query heads=8, KV heads=1, head dimension=256, FP16, NZ KV cache, **physical block size=64**. The model's separate logical block size is not an operator input here.

For query row 123:

1. Set all Q/K/V to zero.
2. Run attention and record the row output.
3. Change only the 14 V tokens immediately after that row's causal cutoff to 64.
4. Run attention again. The row must remain exactly zero.
5. Check the last query row sees the mutation, to verify KV writes actually took effect.

## Expected / observed

| Context | Causal cutoff | Legacy row before → after | Compressed row before → after |
| --- | --- | --- | --- |
| 32768 | 32507 | 0 → 0 | 0 → 0 |
| 32774 | 32513 | 0 → 0 | 0 → 0.0275421142578125 |

Both arms return approximately 0.02734375 for the last query after the mutation, as expected. The script exits 1 when it detects the violation. Identical results were obtained on both chips and in both newer-stack images.

A separate randomized comparison also finds this behavior for single and mixed requests. For qLens `[384,1024,512]`, contextLens `[32774,1024,12544]`, the worst sampled compressed error against the correct CPU FP32 reference is `0.00671887`. At row123/head6, a reference shifted 14 future tokens forward matches within `1.93e-5`.

Could you confirm whether an additional alignment constraint or different mask representation is required for this API, or whether this is an operator defect? We have not isolated the internal CANN/ATB implementation line. A server-only WY change cannot correct this standalone failure.
