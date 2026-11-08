#include <torch/library.h>
#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
extern "C" int launch_moe_prescale(void*,void*,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);

at::Tensor moe_prescale(const at::Tensor& x,const at::Tensor& w,
                       const at::Tensor& scale,const at::Tensor& groups) {
  TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 && x.scalar_type()==at::kHalf && x.dim()==2 && x.is_contiguous(),"NPU contiguous FP16 [M,K] required");
  const int64_t m=x.size(0),k=x.size(1);
  TORCH_CHECK((m==8 || m==16 || m==24 || m==32) && w.dim()==3 && w.scalar_type()==at::kChar && w.size(0)==256 && w.size(2)==k,"small routed batch and 256-expert INT8 bank required");
  const int64_t n=w.size(1);
  TORCH_CHECK((k==2048 && n==512) || (k==256 && n==2048),"unsupported projection dimensions");
  TORCH_CHECK(scale.scalar_type()==at::kFloat && scale.dim()==2 && scale.size(0)==256 && scale.size(1)==n && scale.is_contiguous(),"contiguous FP32 per-channel expert scales required");
  TORCH_CHECK(groups.scalar_type()==at::kLong && groups.dim()==1 && groups.numel()==256 && groups.is_contiguous(),"256 cumulative INT64 group ends required");
  for(const auto& t:{w,scale,groups})TORCH_CHECK(t.device()==x.device(),"all inputs must share one NPU");
  auto out=at::empty({m,n},x.options());
  auto workspace=at::empty({((m+15)/16*16*k)+m*32+256+4096},x.options().dtype(at::kByte));
  auto stream=c10_npu::getCurrentNPUStream();
  for(const auto& t:{x,w,scale,groups,out,workspace})c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
  auto aclstream=stream.stream(false);
  void *xp=x.data_ptr(),*wp=w.data_ptr(),*sp=scale.data_ptr(),*gp=groups.data_ptr(),*op=out.data_ptr(),*work=workspace.data_ptr();
  auto handler=[=]() -> int {return launch_moe_prescale(aclstream,xp,wp,sp,gp,op,work,m,k,n,256,1);};
  at_npu::native::OpCommand::RunOpApiV2("MoEPrescale310",handler);
  return out;
}
TORCH_LIBRARY(moe_prescale_310,m){m.def("grouped(Tensor x, Tensor weight, Tensor scale, Tensor groups) -> Tensor");}
TORCH_LIBRARY_IMPL(moe_prescale_310,PrivateUse1,m){m.impl("grouped",&moe_prescale);}
