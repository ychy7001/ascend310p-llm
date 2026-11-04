# SPDX-License-Identifier: Apache-2.0
"""实验worker：仅改接已验证的FwdO同步补丁；可独立叠加已验证的原生WY。"""
import hashlib
import importlib
import os
from pathlib import Path
import torch
from vllm_ascend._310p.worker_310p import NPUWorker310

PRODUCTION_CHUNK_SHA256='ab2b739bb6630ebe9fcb1d7af820f93adffce79668e32247bbb869119a0d6c69'


def install_maskfix():
    module=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
    actual=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual!=PRODUCTION_CHUNK_SHA256:raise RuntimeError('Production chunk source changed: '+actual)
    if hasattr(module,'_original_fwd_o_before_maskfix'):raise RuntimeError('FwdO already replaced')
    torch.ops.load_library('/maskfix/extension/maskfix_fwd_o.so')
    original=torch.ops._C_ascend.chunk_fwd_o
    used=False
    def run(q,k,v,h,scale,g=None,g_gamma=None,cu_seqlens=None,chunk_indices=None,
            chunk_size=64,transpose_state_layout=False):
        nonlocal used
        supported=(q.device.type=='npu' and q.ndim==4 and k.shape==q.shape and v.ndim==4
            and q.shape[-1]==128 and v.shape[-1]==128 and q.shape[-2]>0 and q.shape[-2]%64==0
            and all(x.dtype==torch.float16 and x.device==q.device for x in (q,k,v,h))
            and g is not None and g.dtype==torch.float32 and g.device==q.device
            and g_gamma is None and chunk_size in (None,64) and transpose_state_layout in (None,False))
        op=torch.ops.maskfix_fwd_o.run if supported else original
        result=op(q,k,v,h,scale,g=g,g_gamma=g_gamma,cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,chunk_size=chunk_size,transpose_state_layout=transpose_state_layout)
        if supported and not used:
            used=True
            print('[maskfix] first FwdO call submitted to isolated synchronization patch',flush=True)
        return result
    module._original_fwd_o_before_maskfix=original
    torch.ops._C_ascend.chunk_fwd_o=run
    print('[maskfix] guarded FwdO override installed; production FwdH retained',flush=True)
    if os.environ.get('EXPERIMENT_NATIVE_WY')=='1':
        from native_wy import install
        install('/native-wy/wy-extension/pr17640_wy.so')


class MaskFixWorker(NPUWorker310):
    def init_device(self):
        super().init_device()
        install_maskfix()
