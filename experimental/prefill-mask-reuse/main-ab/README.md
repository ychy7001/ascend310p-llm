# Reproduce the upstream 310P A/B

The pinned pair and acceptance criteria are in [the protocol](../../../docs/upstream/310p-main-ab-plan.md). Stage `build-main.sh`, the GitHub source archive named `main-<full vllm-ascend sha>.tar.gz`, and its `source.sha256` at the experiment root on a2. Build with the upstream 310P Dockerfile. The script uses the Dockerfile default vLLM release pinned by commit, verifies the CANN base digest, and writes `dependencies.txt` and `image-id.txt` when the image is ready. The background wrapper records the build status in `build-release-exit.txt`. If the container cannot fetch GitHub, pass a local bare repository with the upstream `v0.33.0` tag as the first argument; it is mounted read-only and checked against the expected commit.

Stage the remaining scripts in `ab/`, with the following inputs:

- `candidate.patch`: `patches/upstream/310p-legacy-splitfuse-active-mask.patch`
- `bench_http_stream.py`, `bench_server_phases.py`, `check_long_context_needle.py` from `tools/`
- `frozen-model.sha256`: the pinned model config and quantized weight index checksums

Ensure `launch.sh` is executable. `bash wait-and-run.sh` waits for the image, then runs natural routing. If upstream naturally selects the compressed kernel, the controller runs a separately labeled forced-legacy A/B. Both arms have an identical route/shape probe; only arm B carries the candidate diff. The controller holds `/tmp/a310-npu.lock` while serving and records the actual CLI and device samples. Cold runs and mixed parity are separate. Run Graph/MTP admission in separate configurations; the timing script uses eager/MTP disabled.

`isolated-cpu-test.py` is an interim check for environments that lack the pinned upstream runtime. Place it next to the patched test and source tree; it executes the real upstream layout functions with a mocked NPU format cast and reports the focused unit tests.

`cann921-mask-allocation.py` isolates the mask builder on the official 310P CANN 9.2.1 image. Mount the original and patched `attention_mask.py` as `/workbench/upstream.py` and `/workbench/candidate.py`, and the pinned upstream `utils.py` as `/workbench/upstream-utils.py`. It checks the active tile values against the original mask and measures per-call allocated bytes and synchronized builder time. It does not load model weights or time SplitFuse.
