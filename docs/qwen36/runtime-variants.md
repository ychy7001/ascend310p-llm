# Qwen3.6 Runtime Variants

The repository now describes two separate 310P runtime variants in
`deploy/variants/qwen36.json`:

* `tp1-w4a8` is the existing W4A8/C8/MTP path. Its entrypoint and base lock
  remain unchanged.
* `tp2-w8a8-wy` is the PR17640 WY staging candidate. It uses two 310P
  devices, W8A8 weights, no MTP in the first release, and a separate base
  lock and entrypoint.

The variants share deployment conventions and driver mounts, but they do not
share a runtime overlay. This is deliberate: the tested W4A8 assets include
MTP, C8, and MindIEQ components, while the WY candidate requires the 310P
custom OPP and the PR17640 native WY extension. A single unconditional
`start.sh` would make rollback and provenance ambiguous.

## Packaging order

1. Keep the current TP1 W4A8 image as the rollback and business baseline.
2. Package the tested PR17640 WY artifacts into a separately digest-locked
   image using `deploy/image/base-wy-310p.json`.
3. Run startup, cold-prefill, concurrency, PDF proxy, and rollback checks on
   a2 before considering the TP2 variant for production.
4. Evaluate a newer vLLM-Ascend base as a separate matrix. It must pass both
   variants' startup and quality checks before replacing either base lock.

MTP remains enabled and versioned in the TP1 W4A8 variant. It is intentionally
not combined with WY in the first TP2 W8A8 candidate; adding it would create a
second independent source of quality and memory risk.

## Switch timing

The 310P CANN runtime needs a cold interval between different WY and Torch
container images. On a2, an immediate Torch-to-WY restart reproduced a TBE
subprocess initialization failure, while the same WY image started after a
60-second idle interval. The WY systemd unit therefore includes a 60-second
pre-start delay. A rollback procedure must stop the active container, wait for
the same interval, and only then start the other variant.
