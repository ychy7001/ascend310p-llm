"""Single-variable MoE scheduling workers for the frozen production image."""
from pathlib import Path
from vllm_ascend._310p.worker_310p import NPUWorker310


class ControlWorker(NPUWorker310):
    pass


class CandidateWorker(NPUWorker310):
    def init_device(self):
        super().init_device()
        import importlib
        import torch
        import torch_npu
        torch.ops.load_library(str(Path('/experiment/build/libmoe_prescale_torch.so')))

        @torch.library.register_fake('moe_prescale_310::grouped')
        def fake(x, weight, scale, groups):
            return torch.empty((x.shape[0],weight.shape[1]),device=x.device,dtype=x.dtype)

        module=importlib.import_module('vllm_ascend._310p.fused_moe.moe_mlp')
        original=module.quant_apply_mlp
        def small_m(hidden_states,w1,w1_scale,w2,w2_scale,group_list,group_list_type=1):
            if hidden_states.shape[0] not in (8,16,24,32):
                return original(hidden_states,w1,w1_scale,w2,w2_scale,group_list,group_list_type)
            ends=torch.cumsum(group_list,dim=0) if group_list_type==1 else group_list
            gate=torch.ops.moe_prescale_310.grouped(hidden_states,w1,w1_scale,ends)
            act=torch_npu.npu_swiglu(gate)
            return torch.ops.moe_prescale_310.grouped(act,w2,w2_scale,ends)
        module.quant_apply_mlp=small_m
        print('[moe_tiling] small-M grouped scheduling installed; other operators unchanged',flush=True)
