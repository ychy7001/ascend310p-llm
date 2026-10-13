// SPDX-License-Identifier: Apache-2.0
// 路由索引留在设备；复制token量化码/scale并写16零padding行。
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void qwen36_a8_gather_kernel(
 GM_ADDR q,GM_ADDR scale,GM_ADDR indices,GM_ADDR output,GM_ADDR output_scale,
 int64_t routes,int64_t width,int64_t padded_rows){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
 TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,4096);
 auto values=buffer.Get<int8_t>();auto scales=buffer.Get<float>()[512];
 GlobalTensor<int8_t> src,dst;GlobalTensor<float> ss,ds;GlobalTensor<int64_t> ids;
 src.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(q));
 dst.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
 ss.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
 ds.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output_scale));
 ids.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));
 for(int64_t start=GetBlockIdx()*8;start<padded_rows;start+=GetBlockNum()*8){
  for(int64_t lane=0;lane<8;++lane){
   int64_t row=start+lane;float s=0.f;
   if(row<routes){
    int64_t token=ids.GetValue(row);s=ss.GetValue(token);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopy(values,src[token*width],width);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
   }else{Duplicate(values.ReinterpretCast<int16_t>(),static_cast<int16_t>(0),width/2);}
   scales.SetValue(lane,s);
   SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
   DataCopy(dst[row*width],values,width);
   SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }
  SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(ds[start],scales,8);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
 }
}
