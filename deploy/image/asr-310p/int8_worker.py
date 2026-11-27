"""显式文本小批次MLP动态W8A8；原FP16权重/prefill保留，不修改模型文件。"""
import torch
import torch_npu
import torch.nn.functional as F
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm.model_executor.models.qwen3 import Qwen3MLP

original_process=AscendUnquantizedLinearMethod.process_weights_after_loading
original_mlp_forward=Qwen3MLP.forward
prepared={}

def selected(layer):
    prefix=layer.prefix
    return prefix.startswith('language_model.model.layers.') and prefix.endswith(
        ('.mlp.down_proj', '.mlp.gate_up_proj'))

def process(self,layer):
    original_process(self,layer)
    if selected(layer):
        assert layer.weight.dtype==torch.float16 and layer.weight.ndim==2
        cpu=torch_npu.npu_format_cast(layer.weight,2).cpu().float()
        scale=(cpu.abs().amax(dim=1)/127).clamp_min(1e-12)
        weight=(cpu/scale[:,None]).round().clamp(-127,127).to(torch.int8)
        wi=torch_npu.npu_format_cast(weight.to(layer.weight.device),29).transpose(0,1)
        prepared[layer.prefix]=(wi,scale.to(layer.weight.device))

def dynamic(x:torch.Tensor,weight:torch.Tensor,scale:torch.Tensor,fp16:torch.Tensor)->torch.Tensor:
    if x.ndim!=2 or not 1<=x.shape[0]<=8:
        return F.linear(x,fp16)
        announced.add(key)
    q,s=torch_npu.npu_dynamic_quant(x)
    return torch_npu.npu_quant_matmul(q,weight,scale,pertoken_scale=s,
                                    output_dtype=x.dtype)

def fake(x:torch.Tensor,weight:torch.Tensor,scale:torch.Tensor,fp16:torch.Tensor)->torch.Tensor:
    return x.new_empty((x.shape[0],weight.shape[1]))

direct_register_custom_op(op_name='asr_int8_dynamic',op_func=dynamic,
                          fake_impl=fake,mutates_args=[],dispatch_key='PrivateUse1')

def mlp_forward(self,x):
    if selected(self.down_proj):
        assert self.down_proj.weight.shape==(2048,6144)
        weight,scale=prepared[self.gate_up_proj.prefix]
        intermediate=torch.ops.vllm.asr_int8_dynamic(x,weight,scale,self.gate_up_proj.weight)
        intermediate=self.act_fn(intermediate)
        weight,scale=prepared[self.down_proj.prefix]
        return torch.ops.vllm.asr_int8_dynamic(intermediate,weight,scale,self.down_proj.weight)
    return original_mlp_forward(self,x)

AscendUnquantizedLinearMethod.process_weights_after_loading=process
Qwen3MLP.forward=mlp_forward

class Int8WorkerExtension:
    """Apply the fixed 310P int8_mlp path in the vLLM worker."""
