# SPDX-License-Identifier: Apache-2.0
"""可选TP1原型：Prefill采用MindIEQ grouped，Decode维持native路径。"""
import os
import torch
import torch_npu
from mindieq_activation import quantize_shared_routes
from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE, build_grouped_expert_dispatch

torch.ops.load_library(os.environ['QWEN36_MINDIEQ_LIBRARY'])
original=W4SparseMoE._forward_grouped
reported=False
min_tokens=int(os.environ.get("QWEN36_MINDIEQ_MIN_TOKENS","17"))
if min_tokens<17:
    raise ValueError("QWEN36_MINDIEQ_MIN_TOKENS must be at least 17")
custom_quant=os.environ.get("QWEN36_MINDIEQ_CUSTOM_QUANT","0")=="1"

shared_gate=os.environ.get("QWEN36_MINDIEQ_SHARED_GATE","0")=="1"
if shared_gate and not custom_quant:
    raise ValueError("shared Gate/Up quantization requires custom A8 quantization")

def projection(inputs,bank,ends,prepared=None):
    e,n,k2=bank.weight.shape
    if k2*2 not in (512,2048) or n not in (1024,2048):
        raise RuntimeError('MindIEQ prototype requires Qwen3.6 TP1 expert shapes')
    if prepared is not None:
        q,s=prepared
    elif custom_quant:
        q,s=torch.ops.qwen36_mindieq_grouped.quantize(inputs)
    else:
        q,s=torch_npu.npu_dynamic_quant(inputs)
        q=torch_npu.npu_format_cast(q,2).contiguous()
        q=torch.cat([q,torch.zeros((16,q.shape[1]),device=q.device,dtype=q.dtype)])
        s=torch.cat([s.float(),torch.zeros(16,device=s.device,dtype=torch.float32)])
    packed=bank.weight.view(torch.int32).reshape(e,-1)
    # native scale [N/16,K/128,16]；SSZ checkpoint各group重复相同scale。
    scale=bank.weight_scale.reshape(e,n//16,k2//64,16)[:,:,0,:].reshape(e,n).float().contiguous()
    return torch.ops.qwen36_mindieq_grouped.matmul(q,s,packed,scale,ends)

def forward(self,block_input,weights,ids):
    global reported
    if block_input.shape[0]<min_tokens:
        return original(self,block_input,weights,ids)
    if self.expert_tp_size!=1 or not self.native_int4:
        raise RuntimeError('MindIEQ Prefill prototype currently requires TP1 native W4')
    if not reported:
        print(f"[Qwen36 MindIEQ Prefill] tokens={block_input.shape[0]} native weight storage reused",flush=True)
        reported=True
    result=torch.empty_like(block_input,dtype=self.compute_dtype)
    for start in range(0,block_input.shape[0],self.grouped_chunk_tokens):
        stop=start+self.grouped_chunk_tokens
        inputs=block_input[start:stop];tokens,hidden=inputs.shape
        dispatch=build_grouped_expert_dispatch(weights[start:stop],ids[start:stop],num_local_experts=self.num_local_experts,expert_offset=self.expert_offset,weight_dtype=self.compute_dtype)
        sorted_tokens=dispatch.token_indices.index_select(0,dispatch.order)
        prepared=quantize_shared_routes(inputs,sorted_tokens) if shared_gate else None
        if not shared_gate:
            inputs=inputs.index_select(0,sorted_tokens).contiguous()
        ends=dispatch.group_list.contiguous()
        projected=projection(inputs,self.projections['gate_up_proj'],ends,prepared=prepared).to(self.compute_dtype)
        gate,up=projected.chunk(2,-1)
        activation=(torch.nn.functional.silu(gate)*up).to(self.params_dtype)
        output=projection(activation,self.projections['down_proj'],ends).to(self.compute_dtype)
        output*=dispatch.route_weights.index_select(0,dispatch.order)
        result[start:stop]=output.index_select(0,dispatch.inverse_order).reshape(tokens,self.top_k,hidden).sum(1)
    return result

W4SparseMoE._forward_grouped=forward
print(f'[Qwen36 MindIEQ] optional native-layout TP1 grouped Prefill; custom_quant={custom_quant}; min_tokens={min_tokens}; shared_gate={shared_gate}; native Decode retained',flush=True)
