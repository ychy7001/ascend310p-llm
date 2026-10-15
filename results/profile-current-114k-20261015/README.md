# Current 114K profile attempt

This is a profile attempt against the current stable TP1 service configuration:
W4A8, C8 device-length, MTP1, Decode Graph, prefix cache enabled, 114688 max
model length, and 6 GiB KV budget.

The complete request passed with 114560 prompt tokens and 128 generated tokens.
The request used Graph replay and reported 100% speculative acceptance in the
service log. Wall-clock timing from this profiler run is not a benchmark: the
Torch profiler changed runtime behavior and the 114K prefill took about 274.6 s
while the profiler was active.

CANN profiling data export failed during `stop_profile` with `ERR04005 PROF
internal error` (`CANNExportParser.log`). The raw Torch profiler files are kept
on the A2 deployment host, but no kernel CSV was produced. This result therefore
proves request completion and profiler reproducibility only; it does not replace
the earlier 112K kernel-share profile, whose configuration did not record the
current `QWEN36_C8_DEVICE_LENGTH` setting.

The failure is recorded so future profiling can adjust CANN export settings or
use a shorter/segmented capture rather than treating missing CSV data as zero
cost.
