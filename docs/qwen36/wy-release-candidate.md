# WY Release Candidate

This is the staging release record for the TP2 W8A8 PR17640 WY variant. It is
not a production switch declaration.

## Locked candidate

| Item | Value |
| --- | --- |
| Variant | `tp2-w8a8-wy` |
| Image | `localhost/qwen36-pr17640-wy-locked:20261108` |
| Image ID | `9df3c6f3e42084d77ef258853cf824a011b4d30a1b05acc72a834a691c073163` |
| Base image | `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38` |
| Source | PR17640 snapshot `18163f6b1d208ab3b4fb06e9752225611c37f40e` |
| Custom OPP package | `f00ed7e47310560bcf90fd899947362467ef1d75e73baa2b8ef5b634abe7db10` |
| `vllm_ascend_C` | `c54172cf6f103213812f97688dc57d330b7d271b6df0750f492908b76368f521` |
| Tensor parallel | 2 |
| Graph mode | `FULL_DECODE_ONLY`, capture `[1,2]` |
| MTP | Disabled in the first WY release |

## Verified

- The locked registry base was rebuilt on a2 with the repository-generated
  context and the image default entrypoint.
- Dual-card health and completion smoke passed; the container exited cleanly.
- Earlier equivalent candidate measurements show 4K/8K/16K/32K cold Prefill
  improvement of `30.54%/28.96%/28.64%/27.87%` against the Torch arm.
- Concurrency 1/4/8 and PDF repeatability checks passed on the equivalent
  PR17640 candidate.
- a2 has no residual NPU process after the checks. a1 was not changed.
- FUNSD labeled smoke (20 images, 183 non-empty linked pairs in the evaluated
  subset) produced matching JSON-valid counts of `18/20` for Torch and WY,
  matching field presence of `121/183`, and answer exact-match counts of
  `62/183` versus `64/183`. This is a non-regression smoke signal, not the
  final invoice quality gate.
- SROIE receipt smoke (20 images, four labeled fields per image) produced
  valid JSON and complete field presence for both arms: `20/20` documents and
  `80/80` fields. Field exact-match was `69/80` for Torch and `71/80` for WY.
  The frozen manifest is outside this repository at
  `/home/ychy/workspace/tasks/data/ocr-public/sroie-test20/manifest.json`,
  with manifest SHA256
  `0a3e0eff7ff4bde90c5e0f515a06e0a240b59644c29c229774ab1c16db707ffe`.
  This is a receipt-specific non-regression smoke signal, not a production
  quality gate.
- The expanded SROIE staging set (100 images, 400 labeled fields) produced
  valid JSON `100/100` and complete field presence `400/400` for both arms.
  Overall field exact-match was `347/400` for Torch and `345/400` for WY;
  date plus total exact-match was `194/200` versus `193/200`, and the count of
  mismatched numeric fields was `6` versus `7`. This is within the provisional
  one-point engineering delta on this smoke set, but it is still not a
  manually reviewed production gate.

## Release procedure

1. Stop the active container and remove it.
2. Confirm the NPU is idle and both service states are inactive.
3. Wait at least 60 seconds before starting the other variant. This interval
   is required by the observed 310P CANN TBE initialization behavior.
4. Start the selected variant and require `/health` plus one valid completion.
5. On rollback, repeat steps 1--4. Never perform an immediate image swap.

## Remaining gates

- The locked image 4K/32K cold Prefill and PDF repeat matrix is complete:
  three cold repeats at each text length and three repeats for PDF-01 through
  PDF-04, with valid outputs and zero prefix-cache hits.
- Add a manually reviewed OCR/receipt set, preferably including the actual
  Chinese invoice fields used by the service. The current PDF text-layer proxy
  has
  `manual_ground_truth=false` and is not a business quality gate.
- Compare field-level exact match, numeric-field exact match, JSON validity,
  truncation, and severe-field errors against the Torch arm.
- Keep the candidate staging-only until the paired quality set passes and the
  release procedure has been exercised with the locked image.

The TP1 W4A8/MTP image remains the production baseline and rollback image.
