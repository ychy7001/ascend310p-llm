"""执行现有真实 GDN 检查；本机旧 AI CPU 关闭 JIT 编译路径。"""
import runpy
import sys
import torch
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)
sys.argv = ["check_pr17640_gdn_310.py", "--source",
    "/work/pr17640/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py",
    "--library", "/work/artifacts/gdn-extension/pr17640_gdn.so",
    "--out", "/work/gdn-native-arm.jsonl"]
runpy.run_path("/work/repo/tools/check_pr17640_gdn_310.py", run_name="__main__")
