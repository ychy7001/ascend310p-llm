"""Isolated TP2 diagnostic: reuse the existing TP1 direct-graph hook."""
import os
import sys
import traceback

try:
    import direct_graph_310
    print('[wy-delivery] direct graph hook loaded; force_updatable=' + os.environ.get('QWEN36_C8_FORCE_UPDATABLE_GRAPH', '0'), flush=True)
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(70)
