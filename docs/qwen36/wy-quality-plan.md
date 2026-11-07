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
