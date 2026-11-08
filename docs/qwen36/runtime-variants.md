# Qwen3.6 Runtime Variants

The repository now describes two separate 310P runtime variants in
`deploy/variants/qwen36.json`:

* `tp1-w4a8` is the existing W4A8/C8/MTP path. It already embeds a pinned
  PR17640 GDN/WY implementation through the independent `pr17640_gdn`
  namespace. Its entrypoint and base lock remain unchanged.
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

MTP remains enabled and versioned in the TP1 W4A8 variant, alongside its
existing PR17640 native WY/GDN adapter. Historical MTP1/2 measurements are
recorded in `mindieq-gdn-experiments-20261013.md`; they do not establish
equivalence to the TP2 candidate binding or current combination acceptance.
The first TP2 W8A8 candidate disables MTP. The 2026-11-08 integration audit
and additional experiments are tracked in `wy-mtp-integration-20261108.md`.

## Switch timing

The 310P CANN runtime needs a cold interval between different WY and Torch
container images. On a2, an immediate Torch-to-WY restart reproduced a TBE
subprocess initialization failure, while the same WY image started after a
60-second idle interval. The WY systemd unit therefore includes a 60-second
pre-start delay. A rollback procedure must stop the active container, wait for
the same interval, and only then start the other variant.

## Delivery scope update — 2026-11-08

Reuse the existing TP1 stable profile and registry digest; the extra W4
Torch/native × MTP0/2 matrix was canceled by scope after the native MTP2
text and PDF smoke. No further W4 experiments are required for this delivery.
Focus on the TP2 W8A8 extension, including its Graph/MTP compatibility and
minimal startup, cold-prefill and PDF acceptance. Register the tested image
with an immutable registry digest and its exact profile. The earlier
no-MTP TP2 candidate does not imply that the MTP extension has passed.
