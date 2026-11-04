# Dense post4 runtime screen v2

Explicit entrypoint reached the start script, but inherited C8 sitecustomize polluted manifest parsing and failed before model loading.

Original status/logs retained. The temporary unit hit systemd start-rate limits during cleanup; that failure is distinct from the primary startup result, and recovery is recorded separately. The compatibility image changes multiple runtime components; no component-level cause or production A/B improvement is established.
