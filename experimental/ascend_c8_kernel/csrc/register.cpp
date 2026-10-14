// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include "torch_kernel_helper.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "aclrtlaunch_c8_page_dequant_kernel.h"
#include "aclrtlaunch_c8_page_scatter_kernel.h"
#include "aclrtlaunch_c8_quant_scatter_kernel.h"
#include "aclrtlaunch_c8_table_dequant_kernel.h"
#include "aclrtlaunch_c8_attention_kernel.h"
#include "aclrtlaunch_c8_qk_cube_kernel.h"
#include "aclrtlaunch_c8_attention_cube_kernel.h"
#include "aclrtlaunch_c8_attention_merge_kernel.h"
#include "aclrtlaunch_c8_cache_to_nz_kernel.h"
#include "aclrtlaunch_c8_cache_to_nz_range_kernel.h"
#include "aclrtlaunch_c8_nz_layout_kernel.h"

namespace qwen36_c8 {
at::Tensor nz_layout(const at::Tensor& x){
  TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 && x.scalar_type()==at::kHalf &&
    x.dim()==2 && x.size(0)==16 && x.size(1)>0 && x.size(1)<=1024 && x.size(1)%16==0 &&
    x.is_contiguous(),"expected contiguous NPU half [16,width], aligned width<=1024");
  int64_t width=x.size(1);
  auto y=at::empty({width/16,16,16},x.options());
  auto stream=c10_npu::getCurrentNPUStream().stream(false);
  void *a=x.data_ptr(),*o=y.data_ptr();
  auto launch=[=]() -> int {return ACLRT_LAUNCH_KERNEL(c8_nz_layout_kernel)(1,stream,a,o,width);};
  at_npu::native::OpCommand::RunOpApi("c8_nz_layout_kernel",launch);return y;
}
void cache_to_nz(const at::Tensor& k,const at::Tensor& v,const at::Tensor& ks,
                 const at::Tensor& vs,at::Tensor& ko,at::Tensor& vo,int64_t block,
                 const c10::optional<at::Tensor>& table=c10::nullopt,
                 const c10::optional<at::Tensor>& context_length=c10::nullopt){
  TORCH_CHECK(k.dim()==3 && v.sizes()==k.sizes() && k.scalar_type()==at::kChar &&
    v.scalar_type()==at::kChar && k.is_contiguous() && v.is_contiguous(),"expected matching contiguous INT8 caches");
  int64_t slots=k.size(0),heads=k.size(1),dim=k.size(2);
  TORCH_CHECK(slots>0 && (block==64 || block==128) && slots%block==0 && heads>0 && heads<=16 &&
    dim>0 && dim%32==0 && heads*dim<=4096,"invalid NZ conversion dimensions");
  TORCH_CHECK(ks.sizes()==at::IntArrayRef({slots,16}) && vs.sizes()==ks.sizes() &&
    ks.scalar_type()==at::kHalf && vs.scalar_type()==at::kHalf && ks.is_contiguous() && vs.is_contiguous(),"invalid scales");
  int64_t out_blocks=slots/block;
  if(table){
    TORCH_CHECK(table->device()==k.device() && table->scalar_type()==at::kInt &&
      table->dim()==1 && table->numel()>0 && table->is_contiguous(),"invalid selected page table");
    out_blocks=table->numel();
  }
  TORCH_CHECK(ko.scalar_type()==at::kHalf && vo.scalar_type()==at::kHalf &&
    ko.sizes()==at::IntArrayRef({out_blocks,heads*dim/16,block,16}) && vo.sizes()==ko.sizes() &&
    ko.is_contiguous() && vo.is_contiguous(),"expected contiguous raw NZ output cache shape");
  TORCH_CHECK(k.device().type()==c10::DeviceType::PrivateUse1,"requires NPU");
  for(const auto& t:{v,ks,vs,ko,vo})TORCH_CHECK(t.device()==k.device(),"requires one NPU");
  if(context_length){
    TORCH_CHECK(table.has_value() && context_length->device()==k.device() &&
      context_length->scalar_type()==at::kInt && context_length->numel()==1 &&
      context_length->is_contiguous(),"device context length requires selected pages and one contiguous INT32 value");
  }
  auto stream=c10_npu::getCurrentNPUStream().stream(false);
  void *a=k.data_ptr(),*b=v.data_ptr(),*c=ks.data_ptr(),*d=vs.data_ptr(),*e=ko.data_ptr(),*f=vo.data_ptr();
  void *mapping=table?table->data_ptr():nullptr;bool selected=table.has_value();
  void *length=context_length?context_length->data_ptr():nullptr;bool limited=context_length.has_value();
  auto launch=[=]() -> int {return ACLRT_LAUNCH_KERNEL(c8_cache_to_nz_kernel)(8,stream,a,b,c,d,e,f,
    out_blocks*block,heads,dim,block,mapping,slots/block,selected,length,limited);};
  at_npu::native::OpCommand::RunOpApi("c8_cache_to_nz_kernel",launch);
}
void cache_to_nz_range(const at::Tensor& k,const at::Tensor& v,const at::Tensor& ks,
                      const at::Tensor& vs,at::Tensor& ko,at::Tensor& vo,int64_t block,
                      const at::Tensor& table,int64_t dst_block){
  TORCH_CHECK(k.dim()==3 && v.sizes()==k.sizes() && k.scalar_type()==at::kChar &&
    v.scalar_type()==at::kChar && k.is_contiguous() && v.is_contiguous(),
    "expected matching contiguous INT8 caches");
  int64_t slots=k.size(0),heads=k.size(1),dim=k.size(2);
  TORCH_CHECK(slots>0 && (block==64 || block==128) && slots%block==0 && heads>0 && heads<=16 &&
    dim>0 && dim%32==0 && heads*dim<=4096,"invalid NZ range dimensions");
  TORCH_CHECK(ks.sizes()==at::IntArrayRef({slots,16}) && vs.sizes()==ks.sizes() &&
    ks.scalar_type()==at::kHalf && vs.scalar_type()==at::kHalf && ks.is_contiguous() && vs.is_contiguous(),
    "invalid scales");
  int64_t blocks=slots/block;
  TORCH_CHECK(table.device()==k.device() && table.scalar_type()==at::kInt && table.dim()==1 &&
    table.numel()==1 && table.is_contiguous(),"range conversion requires one page table entry");
  TORCH_CHECK(dst_block>=0 && dst_block<blocks && ko.scalar_type()==at::kHalf && vo.scalar_type()==at::kHalf &&
    ko.sizes()==at::IntArrayRef({blocks,heads*dim/16,block,16}) && vo.sizes()==ko.sizes() &&
    ko.is_contiguous() && vo.is_contiguous(),"expected full contiguous raw NZ output cache shape");
  TORCH_CHECK(k.device().type()==c10::DeviceType::PrivateUse1,"requires NPU");
  for(const auto& t:{v,ks,vs,ko,vo,table})TORCH_CHECK(t.device()==k.device(),"requires one NPU");
  auto stream=c10_npu::getCurrentNPUStream().stream(false);
  void *a=k.data_ptr(),*b=v.data_ptr(),*c=ks.data_ptr(),*d=vs.data_ptr(),*e=ko.data_ptr(),*f=vo.data_ptr(),*m=table.data_ptr();
  auto launch=[=]() -> int {return ACLRT_LAUNCH_KERNEL(c8_cache_to_nz_range_kernel)(8,stream,a,b,c,d,e,f,
    block,heads,dim,block,m,blocks,dst_block);};
  at_npu::native::OpCommand::RunOpApi("c8_cache_to_nz_range_kernel",launch);
}
at::Tensor qk_cube(const at::Tensor& q, const at::Tensor& k, bool transpose_b) {
  TORCH_CHECK(q.dim() == 2 && q.size(0) == 16 && (q.size(1) == 16 || q.size(1) == 128 || q.size(1) == 256) &&
              k.sizes() == q.sizes() && q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
              q.device().type() == c10::DeviceType::PrivateUse1 && q.device() == k.device() &&
              q.is_contiguous() && k.is_contiguous(), "expected same-device contiguous half [16,16/128/256]");
  auto y=at::empty({16,16}, q.options().dtype(at::kFloat));
  auto stream=c10_npu::getCurrentNPUStream().stream(false);
  void *a=q.data_ptr(), *b=k.data_ptr(), *o=y.data_ptr(); int64_t dim=q.size(1);
  auto launch=[=]() -> int { return ACLRT_LAUNCH_KERNEL(c8_qk_cube_kernel)(1,stream,a,b,o,dim,transpose_b); };
  at_npu::native::OpCommand::RunOpApi("c8_qk_cube_kernel",launch);
  return y;
}

at::Tensor attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                    const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& table,
                    int64_t block_size, int64_t length, bool cube = false, int64_t splits = 1) {
  TORCH_CHECK(splits>=1 && splits<=32 && (cube || splits==1), "Split-KV requires Cube and 1..32 splits");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.sizes() == k.sizes(), "expected QHD query and matching K/V cache");
  int64_t queries=q.size(0), heads=q.size(1), dim=q.size(2), kv=k.size(1);
  TORCH_CHECK(queries > 0 && queries <= 8 && heads > 0 && heads <= 64 && kv > 0 && kv <= 16 &&
              heads % kv == 0 && (dim == 128 || dim == 256) && k.size(2) == dim &&
              (block_size == 64 || block_size == 128) && k.size(0) % block_size == 0 &&
              length >= queries && table.dim() == 1 && length <= table.numel() * block_size,
              "unsupported fused C8 attention shape");
  TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1 && q.scalar_type() == at::kHalf &&
              k.scalar_type() == at::kChar && v.scalar_type() == at::kChar &&
              ks.scalar_type() == at::kHalf && vs.scalar_type() == at::kHalf &&
              table.scalar_type() == at::kInt && ks.dim() == 2 && ks.size(0) == k.size(0) &&
              ks.size(1) == 16 && vs.sizes() == ks.sizes(), "invalid fused C8 dtypes or scales");
  for (const auto& t : {q,k,v,ks,vs,table})
    TORCH_CHECK(t.device() == q.device() && t.is_contiguous(), "inputs must be contiguous on one NPU");
  auto y = at::empty(q.sizes(), q.options().dtype(at::kFloat));
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  void *a=q.data_ptr(), *b=k.data_ptr(), *c=v.data_ptr(), *d=ks.data_ptr(),
       *e=vs.data_ptr(), *f=table.data_ptr(), *o=y.data_ptr();
  int64_t blocks=k.size(0)/block_size;
  TORCH_CHECK(!cube || heads / kv <= 16, "Cube prototype supports at most 16 GQA query heads per KV head");
  auto partial=splits>1 ? at::empty({splits,queries,heads,dim+32},q.options().dtype(at::kFloat)) : y;
  void *partials=partial.data_ptr();
  auto launch = [=]() -> int {
    if (cube){
      int rc=ACLRT_LAUNCH_KERNEL(c8_attention_cube_kernel)(queries*kv*splits, stream, a,b,c,d,e,f,
        splits==1?o:partials, queries, heads, kv, dim, block_size, blocks, length, splits);
      if(rc || splits==1)return rc;
      return ACLRT_LAUNCH_KERNEL(c8_attention_merge_kernel)(queries*heads,stream,partials,o,queries*heads,dim,splits);
    }
    return ACLRT_LAUNCH_KERNEL(c8_attention_kernel)(queries*heads, stream, a,b,c,d,e,f,o,
      queries, heads, kv, dim, block_size, blocks, length);
  };
  at_npu::native::OpCommand::RunOpApi(cube ? "c8_attention_cube_kernel" : "c8_attention_kernel", launch);
  return y;
}

at::Tensor table_dequant(const at::Tensor& x, const at::Tensor& scale,
                        const at::Tensor& table, int64_t logical_page,
                        int64_t block_size, int64_t tokens, int64_t heads,
                        int64_t dim) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
              x.device() == scale.device() && x.device() == table.device(),
              "C8 cache and block table must share an NPU");
  TORCH_CHECK(x.scalar_type() == at::kChar && scale.scalar_type() == at::kHalf &&
              table.scalar_type() == at::kInt && x.is_contiguous() &&
              scale.is_contiguous() && table.is_contiguous() && table.dim() == 1,
              "expected contiguous int8 cache, half scales, int32 block table");
  TORCH_CHECK(block_size > 0 && block_size % 16 == 0 && tokens > 0 &&
              tokens <= block_size && tokens <= 4096 && heads > 0 && heads <= 16 &&
              dim > 0 && dim % 32 == 0 && heads * dim <= 32768 &&
              logical_page >= 0 && logical_page < table.numel(),
              "invalid C8 page dimensions or logical page");
  int64_t row = heads * dim;
  TORCH_CHECK(x.numel() > 0 && x.numel() % (block_size * row) == 0 &&
              scale.numel() == (x.numel() / row) * 16, "invalid cache capacity");
  int64_t blocks = x.numel() / (block_size * row);
  auto y = at::empty({tokens, heads, dim}, scale.options());
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  void *a=x.data_ptr(), *s=scale.data_ptr(), *m=table.data_ptr(), *o=y.data_ptr();
  auto launch = [=]() -> int {
    return ACLRT_LAUNCH_KERNEL(c8_table_dequant_kernel)(tokens, stream, a, s, m, o,
      logical_page, block_size, blocks, tokens, heads, dim);
  };
  at_npu::native::OpCommand::RunOpApi("c8_table_dequant_kernel", launch);
  return y;
}

at::Tensor dequant(const at::Tensor& x, const at::Tensor& scale,
                   int64_t tokens, int64_t heads, int64_t dim) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
              x.scalar_type() == at::kChar && scale.scalar_type() == at::kHalf &&
              x.is_contiguous() && scale.is_contiguous(), "invalid C8 inputs");
  TORCH_CHECK(x.numel() == tokens * heads * dim && scale.numel() == tokens * 16,
              "shape does not match runtime sizes");
  TORCH_CHECK(tokens > 0 && tokens <= 4096 && heads > 0 && heads <= 16 && dim > 0 &&
              heads * dim <= 32768 && dim % 32 == 0,
              "requires 1..4096 tokens, 1..16 heads and dim multiple of 32");
  TORCH_CHECK(x.device() == scale.device(), "inputs must share a device");
  auto y = at::empty({tokens, heads, dim}, scale.options());
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  void *a=x.data_ptr(), *s=scale.data_ptr(), *o=y.data_ptr();
  auto launch = [=]() -> int { return ACLRT_LAUNCH_KERNEL(c8_page_dequant_kernel)(tokens, stream, a, s, o, tokens, heads, 16, dim); };
  at_npu::native::OpCommand::RunOpApi("c8_page_dequant_kernel", launch);
  return y;
}

void quant_scatter(const at::Tensor& key,const at::Tensor& value,
                   const at::Tensor& slots,at::Tensor& kc,at::Tensor& vc,
                   at::Tensor& ks,at::Tensor& vs) {
  TORCH_CHECK(key.dim()==3 && key.scalar_type()==at::kHalf &&
              value.sizes()==key.sizes() && value.scalar_type()==at::kHalf,
              "fused C8 write requires matching FP16 [tokens,heads,dim]");
  int64_t tokens=key.size(0),heads=key.size(1),dim=key.size(2);
  TORCH_CHECK(tokens>0 && tokens<=4096 && heads>0 && heads<=16 && (dim==128 || dim==256),
              "fused C8 write supports tokens 1..4096, heads 1..16, dim128/256");
  TORCH_CHECK(kc.dim()==3 && kc.size(1)==heads && kc.size(2)==dim &&
              vc.sizes()==kc.sizes() && ks.dim()==2 && ks.size(0)==kc.size(0) &&
              ks.size(1)==16 && vs.sizes()==ks.sizes(),"invalid fused C8 cache shapes");
  TORCH_CHECK(kc.scalar_type()==at::kChar && vc.scalar_type()==at::kChar &&
              ks.scalar_type()==at::kHalf && vs.scalar_type()==at::kHalf &&
              slots.scalar_type()==at::kInt && slots.numel()==tokens,"invalid fused C8 dtypes/slots");
  TORCH_CHECK(key.device().type()==c10::DeviceType::PrivateUse1,"fused C8 write requires NPU");
  const at::Tensor* tensors[]={&key,&value,&slots,&kc,&vc,&ks,&vs};
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(tensor->device()==key.device() && tensor->is_contiguous(),
                "fused C8 tensors must be contiguous on the same NPU");
  }
  void *k=key.data_ptr(),*v=value.data_ptr(),*s=slots.data_ptr();
  void *ko=kc.data_ptr(),*vo=vc.data_ptr(),*kso=ks.data_ptr(),*vso=vs.data_ptr();
  int64_t capacity=kc.size(0);
  auto stream=c10_npu::getCurrentNPUStream().stream(false);
  int64_t blocks=std::min<int64_t>(tokens*2,32);
  auto launch=[=]() -> int {
    return ACLRT_LAUNCH_KERNEL(c8_quant_scatter_kernel)(blocks,stream,k,v,s,ko,vo,kso,vso,tokens,heads,dim,capacity);
  };
  at_npu::native::OpCommand::RunOpApi("c8_quant_scatter_kernel",launch);
}

void scatter(const at::Tensor& x, const at::Tensor& scale,
             const at::Tensor& slots, at::Tensor& cache,
             at::Tensor& scale_cache, int64_t tokens, int64_t heads,
             int64_t dim) {
  int64_t width = heads * dim;
  TORCH_CHECK(tokens > 0 && tokens <= 4096 && heads > 0 && heads <= 16 &&
              dim > 0 && dim % 32 == 0 && width <= 32768,
              "invalid C8 scatter dimensions");
  TORCH_CHECK(cache.numel() % width == 0 &&
              scale_cache.numel() == (cache.numel() / width) * 16,
              "cache must pair each token slot with a 16-half scale row");
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
              scale.device() == x.device() && slots.device() == x.device() &&
              cache.device() == x.device() && scale_cache.device() == x.device(),
              "all C8 scatter tensors must be on the same NPU");
  TORCH_CHECK(x.scalar_type() == at::kChar && scale.scalar_type() == at::kHalf &&
              slots.scalar_type() == at::kInt && cache.scalar_type() == at::kChar &&
              scale_cache.scalar_type() == at::kHalf, "invalid C8 scatter dtypes");
  TORCH_CHECK(tokens > 0 && x.numel() == tokens * width &&
              scale.numel() == tokens * 16 && slots.numel() == tokens &&
              x.is_contiguous() && scale.is_contiguous() && slots.is_contiguous() &&
              cache.is_contiguous() && scale_cache.is_contiguous(),
              "invalid C8 scatter shapes or strides");
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  void *a=x.data_ptr(), *s=scale.data_ptr(), *m=slots.data_ptr();
  void *c=cache.data_ptr(), *cs=scale_cache.data_ptr();
  int64_t capacity = cache.numel() / width;
  auto launch = [=]() -> int {
    return ACLRT_LAUNCH_KERNEL(c8_page_scatter_kernel)(tokens, stream, a, s, m, c, cs, tokens, heads, 16, width, capacity);
  };
  at_npu::native::OpCommand::RunOpApi("c8_page_scatter_kernel", launch);
}
}

TORCH_LIBRARY_FRAGMENT(npu, m) {
  m.def("c8_quant_scatter(Tensor key, Tensor value, Tensor slots, Tensor(a!) kc, Tensor(b!) vc, Tensor(c!) ks, Tensor(d!) vs) -> ()");
  m.def("c8_nz_layout(Tensor x) -> Tensor");
  m.def("c8_cache_to_nz(Tensor k, Tensor v, Tensor ks, Tensor vs, Tensor(a!) ko, Tensor(b!) vo, int block, Tensor? table=None, Tensor? context_length=None) -> ()");
  m.def("c8_cache_to_nz_range(Tensor k, Tensor v, Tensor ks, Tensor vs, Tensor(a!) ko, Tensor(b!) vo, int block, Tensor table, int dst_block) -> ()");
  m.def("c8_qk_cube(Tensor q, Tensor k, bool transpose_b) -> Tensor");
  m.def("c8_attention(Tensor q, Tensor k, Tensor v, Tensor ks, Tensor vs, Tensor table, int block_size, int length, bool cube=False, int splits=1) -> Tensor");
  m.def("c8_table_dequant(Tensor x, Tensor scale, Tensor table, int logical_page, int block_size, int tokens, int heads, int dim) -> Tensor");
  m.def("c8_page_dequant(Tensor x, Tensor scale, int tokens, int heads, int dim) -> Tensor");
  m.def("c8_page_scatter(Tensor x, Tensor scale, Tensor slots, Tensor(a!) cache, Tensor(b!) scale_cache, int tokens, int heads, int dim) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("c8_quant_scatter",TORCH_FN(qwen36_c8::quant_scatter));
  m.impl("c8_nz_layout", TORCH_FN(qwen36_c8::nz_layout));
  m.impl("c8_cache_to_nz", TORCH_FN(qwen36_c8::cache_to_nz));
  m.impl("c8_cache_to_nz_range", TORCH_FN(qwen36_c8::cache_to_nz_range));
  m.impl("c8_qk_cube", TORCH_FN(qwen36_c8::qk_cube));
  m.impl("c8_attention", TORCH_FN(qwen36_c8::attention));
  m.impl("c8_table_dequant", TORCH_FN(qwen36_c8::table_dequant));
  m.impl("c8_page_dequant", TORCH_FN(qwen36_c8::dequant));
  m.impl("c8_page_scatter", TORCH_FN(qwen36_c8::scatter));
}
