# Dense post4 runtime screen v3

Explicit entrypoint and disabled C8 bootstrap loaded both ranks and completed profile_run; no 507015 in this run. KV admission rejected 262144: required8.09GiB, available7.66GiB after workspace reserve. No health or inference requests.

Original status/logs retained. The temporary unit hit systemd start-rate limits during cleanup; that failure is distinct from the primary startup result, and recovery is recorded separately. The compatibility image changes multiple runtime components; no component-level cause or production A/B improvement is established.
