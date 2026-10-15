# Runtime Lineage and Reproducible Build Plan

## Current experimental state

The three hosts currently use the same logical Qwen3.6 W4A8 stack, but the
runtime is not yet a single reproducible image:

1. The base image on `orangepi` is
   `local/vllm-ascend:v0.23.0-310p-openeuler-orangepi-rcfix`, image digest
   `sha256:a9a8d356f62c7bbdf29fd7a44000d2bc930f124d4a48478e086ddb32a1e4c351`.
   It has no registry digest or image labels, so the image build provenance is
   not recoverable from Docker metadata alone.
   Inspection of the retained Git checkouts inside the image recovered the
   actual source baselines: vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
   and vLLM-Ascend `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`
   (v0.23.0 release branch). The vLLM working tree is clean. With the snapshot
   candidate mounted, the Ascend tree differs only in `utils.py` (explicit
   RC detection override) and `patch_mamba_utils.py` (candidate snapshot read
   and comments). The full working-tree diff is retained at
   `results/mtp2-gdn-state-sync-20261015/orangepi/rc-image-working-tree.patch`.
   This recovers source provenance, but does not recover all image build
   steps or the independently mounted kernels and Python overlays.
2. `/work/repo` is bind-mounted at runtime and supplies the Qwen W4 loader,
   C8 adapter, GDN adapter, MindIEQ adapter, RC guards, launch scripts and
   bootstrap imports. The container image and checked-out repository are two
   independent version axes.
3. The GDN/310P reference comparison uses the local checkout
   `/home/ychy/workspace/reference/vllm-ascend-gdn-pr`, currently based on
   commit `39fef3f8f`. It is an experimental source branch derived from
   vLLM-Ascend work; it is not a Huawei binary release and is not identical to
   the RC image source.

These facts explain why a working service can currently be reproduced only by
combining an image tag, host-side artifacts, a runtime overlay and environment
variables. The stable service is preserved while this is reorganized.

## Target layering

The repository will converge on four explicit layers, each with a clean commit
and a machine-readable manifest:

1. `base`: pinned vLLM and vLLM-Ascend source revision, Python/package versions,
   CANN compatibility and the base image digest.
2. `w4a8-qwen36`: Qwen3.6 model loader, ModelSlim/SSZ metadata reader, weight
   packing ABI and numerical tests.
3. `310p-kernels`: C8 KV path, GDN/310P extension, MindIEQ grouped Prefill and
   RC/EP compatibility guards. Each optional path remains independently
   switchable.
4. `service`: launch profiles, graph capture policy, MTP policy, validation
   suite and host-specific device/runtime manifests.

The target artifact is one image per supported architecture/runtime when the
binary ABI requires it, built from the same source commit and tagged with the
repository commit plus CANN/driver/SOC metadata. Host-specific differences
such as `SOC_VERSION`, RC vs EP, driver ABI and device affinity belong in the
runtime profile, not in an undocumented source overlay.

## Required manifests

Every published image and ModelScope release should carry:

- `git commit` and base vLLM/vLLM-Ascend revision;
- image digest and Dockerfile build arguments;
- CANN, driver, firmware, torch-npu, torch and vLLM versions;
- SOC (`ascend310p1`/`ascend310p3`) and runtime mode (`RC`/`EP`);
- hashes of W4 extension, C8, GDN, MindIEQ libraries and OPP trees;
- model source revision, quantization configuration and checkpoint hashes;
- enabled feature flags and graph capture sizes.

## Migration order

The current bind-mounted deployment remains the reference while the following
steps are completed:

1. Export exact source and artifact hashes from a2, clt and orangepi.
2. Add a pinned base checkout and a Dockerfile/build script that installs only
   source-controlled Python overlays and copies compiled artifacts by hash.
3. Build an image without changing model behavior, then compare startup logs,
   operator resolution and 8-question/A-B/A outputs on one EP and one RC host.
4. Move the accepted-token/GDN candidate into its own commit and image tag;
   do not mix it into the stable MTP1 tag until correctness passes.
5. Publish the image to a registry only after manifests and a rollback tag are
   available. The same image tag can then be reused on compatible hosts, with
   only the host profile selecting RC/EP and CANN-specific settings.

This is a source and packaging correction, not evidence that the current MTP2
state-sync candidate is correct. The candidate remains isolated until its
quality and long-context gates pass.

## Accepted-token candidate status

On 2026-10-15 a temporary RC container mounted only a one-line change in
`patch_mamba_utils.py`: state-index computation used the accepted-count tensor
copied from the current GPU result instead of the mutable InputBatch view.
The container kept the existing W4A8, GDN, MindIEQ and RC image unchanged.

The candidate passed the 8-question and A/B/A short gate on RC with MTP2:
8/8 strict matches, identical reuse output, and 53/53 accepted draft tokens in
the probe window. This is a useful confirmation that the candidate does not
immediately break the path; it is not a long-context or throughput result.
The subsequent 32K attempt was rejected by the benchmark helper with HTTP 400
because the request generator produced a 655,381-token prompt while the
candidate was intentionally started with `MAX_MODEL_LEN=32768`; the model
itself completed startup. The stable RC MTP1 service was restored afterward.

A corrected retry tokenized only a short text unit and repeated its token IDs
to exactly 32,256 input tokens (128 output tokens, max length 32,768). With the
stable container stopped, the same MTP2 eager candidate completed warmup plus
two cold requests: Prefill 507.11/509.80 tok/s, Decode 8.31/8.67 tok/s,
draft acceptance 78.0%/83.3%, and zero prefix-cache hits. After these requests,
the 8-question probe again passed 8/8, A/B/A output was identical, and no canary
leaked. These are correctness/progress observations, not evidence of a speedup
over the MTP1 Graph baseline. See
`results/mtp2-gdn-state-sync-20261015/orangepi/` for raw results and protocol.
The 32,256-token meeting-update canary also passed all four fields (54 output
tokens, Prefill 442.76 tok/s, Decode 9.52 tok/s).

The subsequent Graph candidate (`FULL_DECODE_ONLY`, capture sizes `[1,2,3]`)
completed capture and became healthy, but its first short chat request did not
return. Two Python stack samples showed the engine waiting in
`_get_draft_token_ids_cpu` at `draft_token_ids_event.synchronize()`. The
candidate was terminated without continuing throughput tests, and the stable
MTP1 container was restarted. This locates the wait, not the underlying failing
kernel/stream dependency; no Graph speed or accuracy result exists for this
candidate. The snapshot change alone is insufficient to make RC MTP2 Graph
usable. Next isolate draft Graph from target Graph and inspect the draft-token
copy stream dependency before changing the event protocol.

The follow-up changed only `MTP_DRAFT_EAGER=1` while retaining the snapshot
patch and target Graph. Capture completed (0.21 GiB), but the first short
request timed out after 180 seconds. Both stack samples showed the same wait.
Disabling draft Graph alone is therefore insufficient. The actual copy method
is the Ascend runner override: an NPU copy stream waits on the producer stream,
copies draft IDs to pinned host memory, then records an event. Only the event
reader is inherited from vLLM. The wait does not distinguish an unfinished
producer kernel from a stream dependency or a copy failure. The candidate is
not eligible for performance or deployment recommendations.

A diagnostic follow-up inserted explicit barriers before the draft-copy
producer wait, after the cross-stream wait, and after the copy event. The first
short request failed at the **first producer-stream barrier**, before the copy
was submitted. Runtime error 507018 named an AI CPU `FloorDiv` failure
(stream 34, task 21701, PID 73083). This is stronger evidence than the prior
event-wait stack: an upstream task was already failing in this run. It still
does not identify the Python call that submitted FloorDiv or prove every
historical stall has the same cause. The second-draft slot calculation in
`attn_update_stack_num_spec_norm` is a source candidate to isolate next.
See `results/mtp2-stream-stages-20261015/` for the diagnostic patch and logs.

The next candidate replaced only the second-draft slot quotient/remainder with
exact CPU integer arithmetic, retaining target Graph and draft eager and
removing the diagnostic barriers. Actual positions used int32 with kernel
block size 128. It passed 8/8 and A/B/A before and after 32K requests. Two cold
32,256/128 requests measured Prefill 505.56/507.99 and Decode 20.94/21.89 tok/s,
with zero prefix-cache hits; a 32K meeting-update canary passed all four fields.
Logs confirmed target Graph replay. This is a viable local compatibility
candidate without a driver upgrade, but adds CPU transfers and synchronization,
does not support draft Graph, and has not passed 112K or a matched MTP1 A/B.
The CPU copy can also change ordering, so the experiment does not prove a pure
FloorDiv implementation defect. Keep this patch separate from defaults and
next validate a device-only exact index alternative. Results and patch:
`results/mtp2-slot-cpu-20261015/`.

The bounded device slot arithmetic probe passed 2,097,153 positions and changed-input
Graph replay. With the accepted-count snapshot patch retained, both target and draft
Graph now pass short/reuse quality gates and two cold 32K requests: Prefill
508.39/510.17 tok/s, Decode 23.91/22.90 tok/s. These use the fixed benchmark
and match the draft-eager input hashes; one output has wording differences.
112K, meeting canary and a matched MTP1 comparison remain required before defaulting
to MTP2 on RC. Stable MTP1 was restored and passed its quality gate.
See `results/mtp2-slot-device-20261015/README.md`.

### RC full Graph 112K follow-up (2026-10-15)

Both MTP1 and MTP2 complete the same 114176-token meeting canary with identical
output and pass post-long 8-question/A-B/A gates. Prefill344.26/341.89 tok/s;
short-answer Decode14.58/17.92 tok/s. These single canaries have different
preceding request histories, not a long-Decode throughput comparison. The first
MTP2 benchmark was stopped too early: its warmup completed in 301.78 seconds,
but the next measured request was interrupted;
repeated GDN samples were insufficient evidence of a hang. Record it as interrupted.

At matched112K settings two cold32K requests give MTP1 Decode20.10/20.72 and
MTP2 16.36/16.82 tok/s with identical full outputs. The second draft adds no
accepted tokens in that sequence. Keep RC MTP1 as default and investigate
second-draft state after long Prefill/reuse before promoting patches.
See `results/mtp2-fullgraph-long-20261015/README.md` for logs and limits.
