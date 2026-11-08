"""TP2 legacy SplitFuse Graph compatibility candidate; no kernel changes."""
import os
import sys
import traceback
try:
    import direct_graph_310
    import graph_safe_splitfuse
    graph_safe_splitfuse.install()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(70)
