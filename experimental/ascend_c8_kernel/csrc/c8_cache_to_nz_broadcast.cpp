// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

// Convert physical C8 slots directly to the raw 310P NZ cache ABI, retaining
// the physical block order. Eight cores traverse disjoint, page-local tiles.
extern "C" __global__ __aicore__ void c8_cache_to_nz_kernel(
    GM_ADDR key, GM_ADDR value, GM_ADDR ks, GM_ADDR vs, GM_ADDR kout, GM_ADDR vout,
    int64_t slots, int64_t heads, int64_t dim, int64_t block_size,
    GM_ADDR table, int64_t physical_blocks, bool selected){
  int64_t width=heads*dim;
  int64_t tile=16;
  while(tile*2<=block_size && tile*2<=64 && tile*2*width<=65536)tile*=2;
  TPipe pipe;TBuf<TPosition::VECCALC> ib,hb,sb,bb,zb,gb;
  pipe.InitBuffer(ib,tile*width);pipe.InitBuffer(hb,tile*width*2);pipe.InitBuffer(sb,tile*32);
  pipe.InitBuffer(bb,tile*256*2);pipe.InitBuffer(zb,tile*16*2);pipe.InitBuffer(gb,tile*16*4);
  auto x=ib.Get<int8_t>();auto y=hb.Get<half>();auto scale=sb.Get<half>();
  auto broadcast=bb.Get<half>();auto z=zb.Get<half>();auto offsets=gb.Get<uint32_t>();
  for(int64_t row=0;row<tile;++row)
    for(int64_t lane=0;lane<16;++lane)offsets.SetValue(row*16+lane,(row*width+lane)*2);
  pipe_barrier(PIPE_ALL);
  GlobalTensor<int8_t> inputs[2];GlobalTensor<half> scales[2],outputs[2];
  inputs[0].SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
  inputs[1].SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(value));
  scales[0].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ks));
  scales[1].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vs));
  outputs[0].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(kout));
  outputs[1].SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vout));
  GlobalTensor<int32_t> pages;
  if(selected)pages.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
  for(int64_t t=GetBlockIdx()*tile;t<slots;t+=GetBlockNum()*tile){
    int64_t physical=selected?pages.GetValue(t/block_size):t/block_size;
    bool valid=physical>=0 && physical<physical_blocks;
    int64_t source=physical*block_size+t%block_size;
    for(int side=0;side<2;++side){
      if(valid){
        DataCopy(x,inputs[side][source*width],tile*width);
        DataCopy(scale,scales[side][source*16],tile*16);pipe_barrier(PIPE_ALL);
        Cast(y,x,RoundMode::CAST_NONE,tile*width);pipe_barrier(PIPE_ALL);
        Brcb(broadcast,scale,static_cast<uint8_t>(tile*2),BrcbRepeatParams{1,8});
      }else Duplicate(y,static_cast<half>(0),tile*width);
      pipe_barrier(PIPE_ALL);
      int64_t block=t/block_size,offset=t%block_size;
      for(int64_t col=0;col<width/16;++col){
        Gather(z,y,offsets,static_cast<uint32_t>(col*32),static_cast<uint32_t>(tile*16));
        pipe_barrier(PIPE_V);
        if(valid){
          Mul(z,z,broadcast[(col/(dim/16))*16],128,static_cast<uint8_t>(tile/8),
              BinaryRepeatParams{1,1,16,8,8,128});
        }
        pipe_barrier(PIPE_ALL);
        int64_t dst=((block*(width/16)+col)*block_size+offset)*16;
        DataCopy(outputs[side][dst],z,tile*16);
        pipe_barrier(PIPE_ALL);
      }
      pipe_barrier(PIPE_ALL);
    }
  }
}
