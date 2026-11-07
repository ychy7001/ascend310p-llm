# WY Quality Plan

The existing PDF fixture is useful for startup, repeatability, truncation, and
text-layer proxy checks, but it has no human field labels and cannot be the
production gate.

Freeze a small public OCR subset in two layers:

1. A 20--50 document smoke set covering receipts, forms, multi-line fields,
   dates, amounts, tax identifiers, and low-quality scans.
2. A 200+ document paired set for TP2 W8A8 WY versus the Torch baseline, with
   normalized field-level exact match and numeric-field exact match.

Keep image/PDF bytes, source revision, prompt template, tokenizer revision,
and expected JSON under a manifest with SHA256 values. Dataset license and
redistribution terms must be recorded before mirroring files.

Do not use free-form CER as the release metric. Use amount, date, tax ID,
field presence, JSON validity, truncation, and severe-error counts as business
metrics; retain CER only as a diagnostic signal.

| Stage | TP1 W4A8 | TP2 W8A8 WY | Required result |
| --- | --- | --- | --- |
| Smoke | 20--50 docs | same docs | no crash, valid JSON, no new severe field failures |
| Paired regression | 200+ docs | same docs | report per-field deltas and confidence intervals |
| Staging | fixed replay | fixed replay | rollback and repeatability pass |

The TP2 WY variant remains staging-only until the paired set is available.

The first public labeled subset is now frozen outside the repository:

- Dataset: FUNSD `testing_data`
- Subset: 20 images, 336 linked question/answer pairs
- Archive SHA256: `c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f`
- Subset manifest SHA256: `418c585a3da8cb368f0ff243ac5bf891d24e793cd476e4f8fdf710c213cecb4f`
- Local manifest: `/home/ychy/workspace/tasks/data/ocr-public/funsd-test20/manifest.json`

This subset is suitable for the first paired smoke comparison. It is a form
understanding set rather than a receipt-only set, so it does not replace a
later invoice/receipt set with amount and tax-field coverage.

The first paired run completed on the locked WY image and the Torch arm. Both
arms produced parseable JSON for 18/20 images and exposed 121/183 expected
fields. Answer exact-match counts were Torch 62/183 and WY 64/183. The result
supports “no observed regression on this smoke subset”; it is too small and
too form-oriented to authorize production by itself.
