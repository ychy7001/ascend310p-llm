# SPDX-License-Identifier: Apache-2.0
import sys
import os
sys.path.insert(0, '/bootstrap')
with open('/bootstrap/sitecustomize.py', 'rb') as source:
    exec(compile(source.read(), '/bootstrap/sitecustomize.py', 'exec'))
sys.path.insert(0, '/c8adapter')
if os.environ.get('QWEN36_C8_ENABLED', '1') == '1':
    import c8_vllm_adapter  # noqa: F401,E402
elif os.environ.get('QWEN36_C8_ALLOW_GRAPH', '0') == '1':
    import direct_graph_310  # noqa: F401,E402

if os.environ.get("QWEN36_GDN_ENABLED", "0") == "1":
    import pr17640_gdn_adapter  # noqa: F401,E402

if os.environ.get("QWEN36_ROUTE_TRACE_PATH"):
    import route_occupancy_trace  # noqa: F401,E402

if os.environ.get("QWEN36_MINDIEQ_ENABLED", "0") == "1":
    import mindieq_prefill_adapter  # noqa: F401,E402

if os.environ.get('QWEN36_SYNC_METADATA', '0') == '1':
    import sync_metadata_310
    sync_metadata_310.install()
