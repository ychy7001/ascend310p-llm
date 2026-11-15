# [Test][310P] Add a standalone causal repro for compressed SplitFuse

Related to #<issue-number>

### What this PR does / why we need it?

Adds `examples/310p_splitfuse_causal_repro.py`, an on-demand diagnostic for the 310P compressed SplitFuse path. It compares the legacy and compressed operators at contexts 2048 and 2049, mutates only future V, and checks that the last row sees the mutation while the target row does not. No model, vLLM import, or server setup is needed.

The example returns a nonzero status if causality fails. It is intentionally not auto-discovered by CI: the affected backend currently fails it. The PR supplies a reproducible check for backend investigation and validation of a future fix.

For reference, 64 is a supported 310P block size, but not the backend default — `get_preferred_block_size` keeps 128 (`vllm_ascend/_310p/attention/attention_v1.py:102-107`). The defect is not specific to 64: a block-128/head-dim-128 variant fails too, in 23 of 384 rows at `context=32774, query=384`. The script pins 64 because the shorter context-2049 reproduction only appears at that geometry.

### Does this PR introduce _any_ user-facing change?

No serving behavior, dispatch, or defaults change. Adds a standalone example only. No release note fragment.

### How was this patch tested?

On both 310P3 chips in an Atlas 300I DUO, with CANN 9.2.1 / torch-npu 2.10.0.post7:

```bash
python examples/310p_splitfuse_causal_repro.py --device 0
python examples/310p_splitfuse_causal_repro.py --device 1
```

| Context | Legacy row after mutation | Compressed row after mutation | Last-row control |
| --- | ---: | ---: | ---: |
| 2048 | 0 | 0 | 0.5 |
| 2049 | 0 | 0.5673828125 | 0.5 |

The two chips gave identical results. Exit status 1 is the expected reproduction on this affected stack. `py_compile`, Ruff check/format and `git diff --check` passed.

This PR is based on `main` at `4b3bd7eb`, the head when it was prepared. The hardware evidence is an operator-only run in a different, retained image (`76ea9d24`), which was not built on this host; it is not a serving test at PR HEAD. Both revisions were read to confirm that they select the compressed operator under the same capability check.

### Why no test under `tests/ut/`

There is no meaningful passing assertion to add yet: the correct expectation for the compressed operator on this stack is a failure, so the check would either be skipped or keep CI red until the backend is fixed. Adding a skipped test now would also not catch a regression. Once the operator behavior is resolved, this file should become a normal regression case — I am happy to relocate it under `tests/ut/` in a follow-up PR, or in this one if you prefer that from the start.

### Checklist

- [x] Boundary / edge-case coverage: adjacent passing and failing lengths and an explicit legacy control.
- [x] Meaningful & reliable tests: causal invariant and last-row KV write control, independent of model outputs.

This PR does not claim to fix the backend defect.
