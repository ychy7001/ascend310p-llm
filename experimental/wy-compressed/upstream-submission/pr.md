# [Test][310P] Add a standalone causal repro for compressed SplitFuse

### What this PR does / why we need it?

Adds `examples/310p_splitfuse_causal_repro.py`, an on-demand diagnostic for the 310P compressed SplitFuse path. It compares the legacy and compressed operators at contexts 2048 and 2049, mutates only future V, and checks that the last row sees the mutation while the target row does not. No model, vLLM import, or server setup is needed.

The example returns a nonzero status if causality fails. It is intentionally not auto-discovered by CI: the affected backend currently fails it. The PR supplies a reproducible check for backend investigation and validation of a future fix.

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

The two chips gave identical results. Exit status 1 is the expected reproduction on this affected stack. `py_compile`, Ruff check/format and `git diff --check` passed. The branch is based on `4b3bd7eb`; hardware evidence is an operator-only run in the retained `76ea9d24` image, not a serving test at PR HEAD.

### Checklist

- [x] Boundary / edge-case coverage: adjacent passing and failing lengths and an explicit legacy control.
- [x] Meaningful & reliable tests: causal invariant and last-row KV write control, independent of model outputs.

This PR does not claim to fix the backend defect. Link the upstream issue here once it is opened.
