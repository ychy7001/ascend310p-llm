// SPDX-License-Identifier: Apache-2.0
// Dynamic per-token/head quantization using the stored FP16 scale, plus scatter.
// Reduction/cast ordering follows the proven 310P W4 activation-pack patterns.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void c8_quant_scatter_kernel(
    GM_ADDR key, GM_ADDR value, GM_ADDR slots, GM_ADDR kc, GM_ADDR vc,
    GM_ADDR ksc, GM_ADDR vsc, int64_t tokens, int64_t heads,
    int64_t dim, int64_t capacity) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> ib, fb, tb, qb, nb, hb, ob, rb, gb, mb, xb, sb;
  pipe.InitBuffer(ib, 256*sizeof(half));
  pipe.InitBuffer(fb, 256*sizeof(float));
  pipe.InitBuffer(tb, 256*sizeof(float));
  pipe.InitBuffer(qb, 256*sizeof(float));
  pipe.InitBuffer(nb, 256*sizeof(int32_t));
  pipe.InitBuffer(hb, 256*sizeof(half));
  pipe.InitBuffer(ob, 256*sizeof(int8_t));
  pipe.InitBuffer(rb, 64*sizeof(float));
  pipe.InitBuffer(gb, 64*sizeof(float));
  pipe.InitBuffer(mb, 8*sizeof(float));
  pipe.InitBuffer(xb, 64*sizeof(uint32_t));
  pipe.InitBuffer(sb, 16*sizeof(half));
  auto input=ib.Get<half>(); auto values=fb.Get<float>();
  auto temp=tb.Get<float>(); auto quant=qb.Get<float>();
  auto integers=nb.Get<int32_t>(); auto rounded=hb.Get<half>();
  auto output=ob.Get<int8_t>(); auto pairs=rb.Get<float>();
  auto gathered=gb.Get<float>(); auto maximum=mb.Get<float>();
  auto indices=xb.Get<uint32_t>(); auto scales=sb.Get<half>();
  for (int i=0;i<64;++i) indices.SetValue(i,(i<dim/64?i:0)*2*sizeof(float));
  SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
  GlobalTensor<int32_t> mapping;
  mapping.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slots));
  GlobalTensor<half> src, dst_scale;
  GlobalTensor<int8_t> dst;
  // K and V are independent work items; all heads of one row share one writer.
  for (int64_t unit=GetBlockIdx();unit<tokens*2;unit+=GetBlockNum()) {
    int64_t token=unit/2; bool is_value=unit%2;
    int32_t slot=mapping.GetValue(token);
    if (slot<0 || slot>=capacity) continue;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(is_value?value:key));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(is_value?vc:kc));
    dst_scale.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(is_value?vsc:ksc));
    Duplicate(scales,static_cast<half>(0),16);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (int64_t head=0;head<heads;++head) {
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      DataCopy(input,src[(token*heads+head)*dim],dim);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(values,input,RoundMode::CAST_NONE,dim); PipeBarrier<PIPE_V>();
      Abs(temp,values,dim); PipeBarrier<PIPE_V>();
      WholeReduceMax(pairs,temp,64,dim/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
      PipeBarrier<PIPE_V>();
      Gather(gathered,pairs,indices,static_cast<uint32_t>(0),64); PipeBarrier<PIPE_V>();
      WholeReduceMax(maximum,gathered,dim/64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
      SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
      float scale=maximum.GetValue(0)/127.0f;
      if (scale<0.00006103515625f) scale=0.00006103515625f;
      half stored=static_cast<half>(scale);
      scales.SetValue(head,stored);
      float actual=static_cast<float>(stored);
      SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
      Duplicate(temp,actual,dim); PipeBarrier<PIPE_V>();
      Div(quant,values,temp,dim); PipeBarrier<PIPE_V>();
      Cast(integers,quant,RoundMode::CAST_RINT,dim); PipeBarrier<PIPE_V>();
      Cast(quant,integers,RoundMode::CAST_NONE,dim); PipeBarrier<PIPE_V>();
      Mins(quant,quant,127.0f,dim); PipeBarrier<PIPE_V>();
      Maxs(quant,quant,-127.0f,dim); PipeBarrier<PIPE_V>();
      Cast(rounded,quant,RoundMode::CAST_NONE,dim); PipeBarrier<PIPE_V>();
      Cast(output,rounded,RoundMode::CAST_NONE,dim);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(dst[(static_cast<int64_t>(slot)*heads+head)*dim],output,dim);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(dst_scale[static_cast<int64_t>(slot)*16],scales,16);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }
}
