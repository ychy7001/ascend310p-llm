# SPDX-License-Identifier: Apache-2.0
"""固定9.2.1的配套H/O质量候选；保留原Torch WY，不修改镜像源码。"""
import hashlib
import importlib
import json
import os
from pathlib import Path
from vllm_ascend._310p.worker_310p import NPUWorker310


class CANN921GDNPairWorker(NPUWorker310):
    def init_device(self):
        import torch
        from vllm_ascend.utils import enable_custom_op
        assert enable_custom_op()
        root = Path('/gdn-pair')
        vendor = root/'opp/vendors/hyperqwen_gdn_pair921_transformer'
        paths = [str(vendor)] + [x for x in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':')
                                 if x and x != str(vendor)]
        os.environ['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(paths)
        # Match the operator gate: bootstrap and vendor order before ACL init.
        super().init_device()
        module = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != '4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f':
            raise RuntimeError('Unexpected9.2.1 chunk source')
        if hasattr(module, '_production_wy_before_native') or os.environ.get('EXPERIMENT_NATIVE_WY') == '1':
            raise RuntimeError('Expected original Torch WY')
        artifacts = json.loads((root/'operator-artifacts.json').read_text())
        for item, expected in artifacts.items():
            if hashlib.sha256((root/item).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Operator artifact changed: '+item)
        vendor = root/'opp/vendors/hyperqwen_gdn_pair921_transformer'
        # NPUBridge captures this list on dlopen; bootstrap must run first.
        paths = [str(vendor)] + [x for x in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':')
                                 if x and x != str(vendor)]
        os.environ['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(paths)
        torch.ops.load_library(str(root/'extension/pr17640_gdn_pair.so'))
        torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h_vllm = torch.ops.pr17640_gdn_pair.fwd_h
        torch.ops._C_ascend.chunk_fwd_o_vllm = torch.ops.pr17640_gdn_pair.fwd_o
        print('[cann921_gdn_pair] '+json.dumps(dict(commit='18163f6b1d208ab3b4fb06e9752225611c37f40e',
              wy='original Torch WY', custom_api_search_paths=paths, operator_artifacts=artifacts)), flush=True)
