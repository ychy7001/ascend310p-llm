// SPDX-License-Identifier: Apache-2.0
// 独立实验：每个 FP16 token 共用一个 INT8 scale，输出原生 INT4 limbs。
// 算法沿用社区整数 limb 编码；仅用于隔离的 per-channel accumulation OPP。
#include "kernel_operator.h"
namespace {
using namespace AscendC;
struct PackTiling { int64_t groups; int64_t groupsPerRow; };
class PackPerToken {
 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR lo, GM_ADDR hi, GM_ADDR sx, GM_ADDR sum,
                             int64_t groups, int64_t groupsPerRow) {
    const int64_t width=groupsPerRow*128, rows=groups/groupsPerRow;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    lo_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(lo));
    hi_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(hi));
    sx_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sx));
    sum_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
    pipe_.InitBuffer(buf_,81920);
    auto input=buf_.Get<half>();
    auto values=buf_.Get<float>()[2048];
    auto quant=buf_.Get<float>()[4096];
    auto temp=buf_.Get<float>()[8192];
    auto integers=buf_.Get<int32_t>()[12288];
    auto limb=buf_.Get<half>()[30720];
    auto packed=buf_.Get<int8_t>()[69632];
    auto pairs=buf_.Get<float>()[18432];
    auto gathered=buf_.Get<float>()[18560];
    auto maximum=buf_.Get<float>()[18624];
    auto totals=buf_.Get<float>()[18848];
    auto indices=buf_.Get<uint32_t>()[18944];
    auto scales=buf_.Get<float>()[19200];
    auto sums=buf_.Get<float>()[19456];
    for(uint32_t i=0;i<64;++i) indices.SetValue(i,(i<width/64?i:0)*2*sizeof(float));
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    for(int64_t row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      DataCopy(input,x_[row*width],width);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(values,input,RoundMode::CAST_NONE,width);PipeBarrier<PIPE_V>();
      Abs(temp,values,width);PipeBarrier<PIPE_V>();
      WholeReduceMax(pairs,temp,64,width/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
      PipeBarrier<PIPE_V>();
      Gather(gathered,pairs,indices,static_cast<uint32_t>(0),64);PipeBarrier<PIPE_V>();
      WholeReduceMax(maximum,gathered,width/64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
      SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
      const float maxValue=maximum.GetValue(0);
      const float scale=maxValue==0.0f?1.0f:maxValue/127.0f;
      Duplicate(temp,scale,width);PipeBarrier<PIPE_V>();
      Div(quant,values,temp,width);PipeBarrier<PIPE_V>();
      Cast(integers,quant,RoundMode::CAST_RINT,width);PipeBarrier<PIPE_V>();
      Cast(quant,integers,RoundMode::CAST_NONE,width);PipeBarrier<PIPE_V>();
      Mins(quant,quant,127.0f,width);PipeBarrier<PIPE_V>();
      Maxs(quant,quant,-127.0f,width);PipeBarrier<PIPE_V>();
      Duplicate(totals,0.0f,32);
      Add(temp,quant,quant[64],64,groupsPerRow,{1,1,1,8,16,16});PipeBarrier<PIPE_V>();
      WholeReduceSum(totals,temp,64,groupsPerRow,1,1,8);PipeBarrier<PIPE_V>();
      Brcb(sums,totals,(groupsPerRow+7)/8,{1,8});
      Duplicate(scales,scale,groupsPerRow*8);PipeBarrier<PIPE_V>();
      Muls(temp,quant,1.0f/16.0f,width);PipeBarrier<PIPE_V>();
      Cast(integers,temp,RoundMode::CAST_FLOOR,width);PipeBarrier<PIPE_V>();
      Cast(temp,integers,RoundMode::CAST_NONE,width);PipeBarrier<PIPE_V>();
      Cast(limb,temp,RoundMode::CAST_NONE,width);PipeBarrier<PIPE_V>();
      Cast(packed.ReinterpretCast<int4b_t>(),limb,RoundMode::CAST_NONE,width);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(hi_[row*width/2],packed,width/2);
      DataCopy(sx_[row*groupsPerRow*8],scales,groupsPerRow*8);
      DataCopy(sum_[row*groupsPerRow*8],sums,groupsPerRow*8);
      Muls(temp,temp,-16.0f,width);PipeBarrier<PIPE_V>();
      Add(temp,quant,temp,width);PipeBarrier<PIPE_V>();
      Adds(temp,temp,-8.0f,width);PipeBarrier<PIPE_V>();
      Cast(limb,temp,RoundMode::CAST_NONE,width);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      Cast(packed.ReinterpretCast<int4b_t>(),limb,RoundMode::CAST_NONE,width);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(lo_[row*width/2],packed,width/2);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
  }
 private:
  TPipe pipe_;TBuf<TPosition::VECCALC> buf_;
  GlobalTensor<half> x_;GlobalTensor<int8_t> lo_,hi_;GlobalTensor<float> sx_,sum_;
};
}
extern "C" __global__ __aicore__ void qwen_w4_a8_pack_v310(GM_ADDR x, GM_ADDR low, GM_ADDR high,
  GM_ADDR scale, GM_ADDR sum, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  auto td=reinterpret_cast<__gm__ PackTiling*>(tiling);
  PackPerToken op;op.Run(x,low,high,scale,sum,td->groups,td->groupsPerRow);
}
