# 310P Graph upstream handoff

Updated 2026-10-13. This note separates the reproducible 310P experiment from
the upstream change that should be reviewed by Huawei/vLLM-Ascend maintainers.

## Observed contract

The pinned experiment image (`vllm-ascend:v0.23.0.post1-310p`) reached
`graph_params=None` in the generic FULL Graph parameter-update path for the
310P attention backend. The experiment therefore enables
`QWEN36_C8_FORCE_UPDATABLE_GRAPH=1`, which makes
`acl_graph.update_full_graph_params()` return through the updatable branch.
With `COMPILATION_MODE=0`, capture/replay passed for C8+MTP1 and C8+MTP2 at
4K, 32K, and 112K on a real Ascend 310P3.

This is a version-specific diagnostic workaround. It must not be copied into
the default backend without checking the exact `vllm-ascend` revision and
attention backend class exposed by the deployment image.

## Pinned image evidence

The check was run in the exact `quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`
image used for the a2 measurements, with Triton disabled and the same custom
extension/OPP bootstrap as the service. It printed:

```text
impl_mro (<class 'vllm_ascend._310p.attention.attention_v1.AscendAttentionBackendImpl310'>,
         <class 'vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl'>,
         <class 'vllm.v1.attention.backend.AttentionImpl'>,
         <class 'vllm.v1.attention.backend.AttentionImplBase'>, ...)
impl_is_backend False
use_updatable_graph_import_error: cannot import name 'use_updatable_graph'
```

The bootstrap also reported a missing optional `qwen36_w4_patch` import while
loading the standalone inspection script; this did not affect the MRO result.
The absence of `use_updatable_graph` confirms that the pinned image predates
the newer upstream marker-based contract. This is why the experiment uses an
explicit compatibility switch rather than claiming that the current image has
the upstream fix.

## Minimal upstream candidate

The intended upstream change is not a global monkeypatch. In the 310P backend,
make the backend identity and Graph lifecycle agree:

```diff
diff --git a/vllm_ascend/_310p/attention/attention_v1.py b/vllm_ascend/_310p/attention/attention_v1.py
@@
 class AscendAttentionBackendImpl310(AscendAttentionBackendImpl):
+    # 310P PA/FIA updates metadata through the updatable Graph path.
+    # Keep this paired with the backend class returned by get_impl_cls().
     ...
```

The exact implementation must be selected against the target upstream
revision: either the backend class must inherit the common
`AscendAttentionBackend` marker, or `use_updatable_graph()` must recognize the
310P backend class. The patch must not simply return early for every 310P
attention implementation; eager, decode, speculative draft, and non-PA
backends need their own lifecycle checks.

## Review gate

Before proposing the patch upstream, run the following on the exact image and
record the output:

```bash
python3 - <<'PY'
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
from vllm_ascend.utils import use_updatable_graph
print(AscendAttentionBackend310.__mro__)
print(issubclass(AscendAttentionBackend310, AscendAttentionBackend))
print(use_updatable_graph(AscendAttentionBackend310))
PY
```

Then repeat the existing 4K sanity, 32K needle, 112K near-limit needle, and
MTP2 capture. A valid upstream patch must remove the environment monkeypatch,
preserve `graph_params` initialization for both target and draft graphs, and
show no regression in eager mode. The current repository does not claim this
upstream patch is complete; it records the smallest code boundary that needs
maintainer confirmation.
