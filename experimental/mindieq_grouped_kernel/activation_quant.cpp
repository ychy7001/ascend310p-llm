// SPDX-License-Identifier: Apache-2.0
// FP16输入，FP32每行absmax/div/round，输出ND INT8及FP32 scale。
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void qwen36_a8_quant_kernel(
 GM_ADDR x,GM_ADDR q,GM_ADDR scale,int64_t rows,int64_t width,int64_t padded_rows){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
 TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,81920);
 auto input=buffer.Get<half>();auto values=buffer.Get<float>()[2048];
 auto quant=buffer.Get<float>()[4096];auto temp=buffer.Get<float>()[8192];
 auto integers=buffer.Get<int32_t>()[12288];auto rounded=buffer.Get<half>()[30720];
 auto output=buffer.Get<int8_t>()[69632];auto pairs=buffer.Get<float>()[18432];
 auto gathered=buffer.Get<float>()[18560];auto maximum=buffer.Get<float>()[18624];
 auto indices=buffer.Get<uint32_t>()[18944];auto scales=buffer.Get<float>()[19200];
 GlobalTensor<half> src;GlobalTensor<int8_t> dst;GlobalTensor<float> ds;
 src.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
 dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(q));
 ds.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
 for(uint32_t i=0;i<64;++i)indices.SetValue(i,(i<width/64?i:0)*2*sizeof(float));
 SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
 for(int64_t start=GetBlockIdx()*8;start<padded_rows;start+=GetBlockNum()*8){
  for(int64_t lane=0;lane<8;++lane){
   int64_t row=start+lane;float s=0.f;
   if(row<rows){
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopy(input,src[row*width],width);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(values,input,RoundMode::CAST_NONE,width);PipeBarrier<PIPE_V>();
    Abs(temp,values,width);PipeBarrier<PIPE_V>();
    WholeReduceMax(pairs,temp,64,width/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
    PipeBarrier<PIPE_V>();Gather(gathered,pairs,indices,static_cast<uint32_t>(0),64);PipeBarrier<PIPE_V>();
    WholeReduceMax(maximum,gathered,width/64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float max_value=maximum.GetValue(0);s=max_value==0.f?1.f:max_value/127.f;
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(temp,s,width);PipeBarrier<PIPE_V>();Div(quant,values,temp,width);PipeBarrier<PIPE_V>();
    Cast(integers,quant,RoundMode::CAST_RINT,width);PipeBarrier<PIPE_V>();
    Cast(quant,integers,RoundMode::CAST_NONE,width);PipeBarrier<PIPE_V>();
    Mins(quant,quant,127.f,width);PipeBarrier<PIPE_V>();Maxs(quant,quant,-127.f,width);PipeBarrier<PIPE_V>();
    Cast(rounded,quant,RoundMode::CAST_NONE,width);PipeBarrier<PIPE_V>();
    Cast(output,rounded,RoundMode::CAST_NONE,width);
   }else{Duplicate(output.ReinterpretCast<int16_t>(),static_cast<int16_t>(0),width/2);}
   scales.SetValue(lane,s);
   SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
   DataCopy(dst[row*width],output,width);
   SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }
  SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(ds[start],scales,8);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
 }
}
