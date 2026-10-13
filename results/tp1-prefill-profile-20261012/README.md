# TP1 W4A8 Prefill Profile

This profile was collected on one Ascend 310P3 with the Qwen3.6-35B-A3B W4A8
native kernel. It is a baseline for kernel tuning, not an end-to-end product
benchmark. The service used the isolated TP1 eager configuration and a 4096
token prefill workload.

## Kernel totals

`kernel_details.csv` contains 400 routed-expert W4 MatMul calls and 400 pack
calls:

| Kernel | Calls | Total time |
| --- | ---: | ---: |
| `QwenW4A8Int4MatmulV310` | 400 | 3980.758 ms |
| `QwenW4A8PackV310` | 400 | 179.440 ms |

The MatMul shape breakdown is:

| Calls | Total time | Input shape prefix |
| ---: | ---: | --- |
| 160 | 2106.300 ms | `5120,1024` |
| 160 | 1196.503 ms | `5120,256` |
| 40 | 432.098 ms | `4096,1024` |
| 40 | 245.857 ms | `4096,256` |

The 5120-row calls correspond to the main routed prefill chunk. The profile
therefore points to the expert MatMul schedule as the first optimization target;
pack, HCCL communication, and host launch time are not the dominant cost in
this run.

## Files

- `kernel_details.csv`: device kernel timing and input shapes
- `task_time.csv`: device task timeline
- `step_trace_time.csv`: step-level compute/communication summary
- `api_statistic.csv`: ACL API aggregation
- `op_statistic.csv`: operator aggregation

The rejected M=64 candidate is recorded separately in
`results/m64-prefill-candidate-20261013/`; its 4096-token median was 376.76
tok/s and did not improve the existing baseline, so the source change was
reverted.
