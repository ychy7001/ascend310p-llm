"""TP2 legacy SplitFuse Graph compatibility; optional lifecycle audit."""
import os
import sys
import traceback
try:
    import direct_graph_310
    import graph_safe_splitfuse
    graph_safe_splitfuse.install()
    if os.environ.get('QWEN36_GRAPH_AUDIT') == '1':
        import graph_lifecycle_audit
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(70)
