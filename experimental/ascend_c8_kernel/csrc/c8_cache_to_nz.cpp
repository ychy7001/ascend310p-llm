// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

// Convert physical C8 slots directly to the raw 310P NZ cache ABI, retaining
// the physical block order. Eight cores traverse disjoint, page-local tiles.
extern "C" __global__ __aicore__ void c8_cache_to_nz_kernel(
    GM_ADDR key, GM_ADDR value, GM_ADDR ks, GM_ADDR vs, GM_ADDR kout, GM_ADDR vout,
    int64_t slots, int64_t heads, int64_t dim, int64_t block_size,
    GM_ADDR table, int64_t physical_blocks, bool selected, GM_ADDR context_length, bool limited){
  TPipe pipe;TBuf<TPosition::VECCALC> ib,hb,sb;
  pipe.InitBuffer(ib,65536);pipe.InitBuffer(hb,65536*2);pipe.InitBuffer(sb,128*32);
  auto x=ib.Get<int8_t>();auto y=hb.Get<half>();auto scale=sb.Get<half>();
  GlobalTensor<int8_t> inputs[2];GlobalTensor<half> scales[2],outputs[2];
  inputs[0].SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
  inputs[1].SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(value));
  scales[0].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ks));
  scales[1].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vs));
  outputs[0].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(kout));
  outputs[1].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vout));
  int64_t width=heads*dim;
  GlobalTensor<int32_t> pages;
  if(selected)pages.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
  int64_t tile=16;
  while(tile*2<=block_size && tile*2*width<=65536)tile*=2;
  // Read the replay-updated length on device. Convert complete live pages;
  // native PA masks the final page. Unused NZ pages retain old values and
  // must never be read beyond this same context length.
  int64_t active_slots=slots;
  if(limited){
    GlobalTensor<int32_t> length;
    length.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(context_length));
    int64_t tokens=length.GetValue(0);
    ASCENDC_ASSERT(tokens>=0 && tokens<=slots, { KERNEL_LOG(KERNEL_ERROR, "invalid C8 context length"); });
    if(tokens<0 || tokens>slots)return;
    active_slots=((tokens+block_size-1)/block_size)*block_size;
  }
  for(int64_t t=GetBlockIdx()*tile;t<active_slots;t+=GetBlockNum()*tile){
    int64_t physical=selected?pages.GetValue(t/block_size):t/block_size;
    bool valid=physical>=0 && physical<physical_blocks;
    int64_t source=physical*block_size+t%block_size;
    for(int side=0;side<2;++side){
      if(valid){
        DataCopy(x,inputs[side][source*width],tile*width);
        DataCopy(scale,scales[side][source*16],tile*16);pipe_barrier(PIPE_ALL);
        Cast(y,x,RoundMode::CAST_NONE,tile*width);pipe_barrier(PIPE_ALL);
        for(int64_t r=0;r<tile;++r)
          for(int64_t h=0;h<heads;++h)
            Muls(y[r*width+h*dim],y[r*width+h*dim],scale.GetValue(r*16+h),dim);
      }else Duplicate(y,static_cast<half>(0),tile*width);
      pipe_barrier(PIPE_ALL);
      int64_t block=t/block_size,offset=t%block_size;
      DataCopyParams nz{static_cast<uint16_t>(tile),1,
                        static_cast<uint16_t>(width/16-1),0};
      for(int64_t col=0;col<width/16;++col){
        int64_t dst=((block*(width/16)+col)*block_size+offset)*16;
        DataCopy(outputs[side][dst],y[col*16],nz);
      }
      pipe_barrier(PIPE_ALL);
    }
  }
}
