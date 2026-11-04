# SPDX-License-Identifier: Apache-2.0
"""在实验 worker 安装已通过门禁的上游 H/O 对；不修改磁盘上的生产模块。"""
import hashlib
import importlib
import os
from pathlib import Path
from vllm_ascend._310p.worker_310p import NPUWorker310


class GDNPairWorker(NPUWorker310):
    def init_device(self):
        super().init_device()
        import torch
        module = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
        from native_wy import PRODUCTION_CHUNK_SHA256
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != PRODUCTION_CHUNK_SHA256:
            raise RuntimeError('Unexpected production chunk module')
        torch.ops.load_library('/gdn-pair/extension/pr17640_gdn_pair.so')
        torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h = torch.ops.pr17640_gdn_pair.fwd_h
        torch.ops._C_ascend.chunk_fwd_o = torch.ops.pr17640_gdn_pair.fwd_o
        if os.environ.get('EXPERIMENT_NATIVE_WY') == '1':
            from native_wy import install
            install()
        print('[gdn_pair] PR17640 18163f6b paired FwdH/FwdO installed; upstream NZ state contract retained; native_wy='+os.environ.get('EXPERIMENT_NATIVE_WY','0'),flush=True)
