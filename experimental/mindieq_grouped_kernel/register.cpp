// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_mindieq_grouped_kernel.h"
at::Tensor grouped(const at::Tensor& a,const at::Tensor& as,const at::Tensor& w,const at::Tensor& ws,const at::Tensor& ends,int64_t minimum_rows){
 TORCH_CHECK(a.device().type()==c10::DeviceType::PrivateUse1 && a.scalar_type()==at::kChar && a.dim()==2 && a.is_contiguous(),"contiguous NPU int8 activation required");
 TORCH_CHECK(w.scalar_type()==at::kInt && w.dim()==2 && w.is_contiguous() && ws.scalar_type()==at::kFloat && ws.dim()==2 && ws.is_contiguous(),"packed int32 [E,N*K/8], fp32 scale [E,N] required");
 int64_t e=w.size(0),k=a.size(1),n=ws.size(1),m=a.size(0)-16;
 TORCH_CHECK(e>0 && e<=256 && k%64==0 && n%16==0 && w.size(1)==n*k/8 && ws.size(0)==e && as.numel()==a.size(0) && as.scalar_type()==at::kFloat && as.is_contiguous() && ends.scalar_type()==at::kLong && ends.numel()==e && ends.is_contiguous() && minimum_rows==0,"invalid grouped contract; A requires 16 padding rows");
 for(const auto& t:{as,w,ws,ends})TORCH_CHECK(t.device()==a.device(),"one NPU required");
 auto o=at::empty({m,n},a.options().dtype(at::kHalf));auto stream=c10_npu::getCurrentNPUStream();
 for(const auto& t:{a,as,w,ws,ends,o})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
 void *ap=a.data_ptr(),*asp=as.data_ptr(),*wp=w.data_ptr(),*wsp=ws.data_ptr(),*ep=ends.data_ptr(),*op=o.data_ptr();auto aclstream=stream.stream(false);
 auto handler=[=]() -> int {return ACLRT_LAUNCH_KERNEL(mindieq_grouped_kernel)(8,aclstream,ap,asp,wp,wsp,ep,op,e,k,n,minimum_rows);};
 at_npu::native::OpCommand::RunOpApiV2("MindIEQGrouped",handler);return o;
}
TORCH_LIBRARY(qwen36_mindieq_grouped,m){m.def("matmul(Tensor a,Tensor ascale,Tensor packed,Tensor wscale,Tensor ends,int minimum_rows=0) -> Tensor");}
TORCH_LIBRARY_IMPL(qwen36_mindieq_grouped,PrivateUse1,m){m.impl("matmul",&grouped);}
