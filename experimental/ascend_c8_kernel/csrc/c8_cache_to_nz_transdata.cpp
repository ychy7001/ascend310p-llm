// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
#ifndef C8_NZ_MAX_TILE
#define C8_NZ_MAX_TILE 16
#endif

// Convert physical C8 slots directly to the raw 310P NZ cache ABI, retaining
// the physical block order. Eight cores traverse disjoint, page-local tiles.
extern "C" __global__ __aicore__ void c8_cache_to_nz_kernel(
    GM_ADDR key, GM_ADDR value, GM_ADDR ks, GM_ADDR vs, GM_ADDR kout, GM_ADDR vout,
    int64_t slots, int64_t heads, int64_t dim, int64_t block_size,
    GM_ADDR table, int64_t physical_blocks, bool selected){
  int64_t width=heads*dim,tile=16;
#ifdef C8_NZ_PIPELINE
  while(tile*2<=C8_NZ_MAX_TILE && tile*2<=block_size && tile*2*width<=16384)tile*=2;
#else
  while(tile*2<=C8_NZ_MAX_TILE && tile*2<=block_size && tile*2*width<=32768)tile*=2;
#endif
  TPipe pipe;TBuf<TPosition::VECCALC> ib,hb,sb,tb,ob;
  pipe.InitBuffer(ib,tile*width);pipe.InitBuffer(hb,tile*width*2);pipe.InitBuffer(sb,tile*32);
  pipe.InitBuffer(tb,tile*width*2);pipe.InitBuffer(ob,tile*width*2);
  auto tmp=tb.Get<half>();auto packed=ob.Get<half>();
  auto x=ib.Get<int8_t>();auto y=hb.Get<half>();auto scale=sb.Get<half>();
#ifdef C8_NZ_PIPELINE
  TBuf<TPosition::VECCALC> ob2;
  pipe.InitBuffer(ob2,tile*width*2);
  auto alternate=ob2.Get<half>();
  auto inputReady=pipe.FetchEventID(HardEvent::MTE2_V);
  auto scalarReady=pipe.FetchEventID(HardEvent::MTE2_S);
  auto outputReady=pipe.FetchEventID(HardEvent::V_MTE3);
  auto inputConsumed=pipe.FetchEventID(HardEvent::V_MTE2);
  auto outputDone0=pipe.FetchEventID(HardEvent::MTE3_V);
  auto outputDone1=pipe.FetchEventID(HardEvent::MTE3_V);
  bool pending[2]={false,false};int buffer=0;
#endif
#ifdef C8_NZ_COMPACT_SCALE
  TBuf<TPosition::VECCALC> gb,cb,bb;
  pipe.InitBuffer(gb,tile*4);pipe.InitBuffer(cb,tile*2);
  pipe.InitBuffer(bb,tile*16*2);
  auto indices=gb.Get<uint32_t>();auto compact=cb.Get<half>();auto broadcast=bb.Get<half>();
  for(int64_t r=0;r<tile;++r)indices.SetValue(r,r*32);
  pipe_barrier(PIPE_ALL);
#endif
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
#ifdef C8_NZ_PIPELINE
      packed=buffer==0?ob.Get<half>():alternate;
      if(pending[buffer]){
        if(buffer==0)WaitFlag<HardEvent::MTE3_V>(outputDone0);
        else WaitFlag<HardEvent::MTE3_V>(outputDone1);
        pending[buffer]=false;
      }
#endif
      if(valid){
        DataCopy(x,inputs[side][source*width],tile*width);
        DataCopy(scale,scales[side][source*16],tile*16);
#ifdef C8_NZ_PIPELINE
        SetFlag<HardEvent::MTE2_V>(inputReady);WaitFlag<HardEvent::MTE2_V>(inputReady);
        SetFlag<HardEvent::MTE2_S>(scalarReady);WaitFlag<HardEvent::MTE2_S>(scalarReady);
#else
        pipe_barrier(PIPE_ALL);
#endif
        Cast(y,x,RoundMode::CAST_NONE,tile*width);pipe_barrier(PIPE_V);
#ifndef C8_NZ_COMPACT_SCALE
        for(int64_t r=0;r<tile;++r)
          for(int64_t h=0;h<heads;++h)
            Muls(y[r*width+h*dim],y[r*width+h*dim],scale.GetValue(r*16+h),dim);
#endif
      }else Duplicate(y,static_cast<half>(0),tile*width);
      pipe_barrier(PIPE_V);
      int64_t block=t/block_size,offset=t%block_size;
      uint64_t sources[16],destinations[16];
      for(int64_t row=0;row<tile;row+=16){
      for(int64_t base=0;base<width/16;base+=128){
        int64_t repeats=width/16-base;if(repeats>128)repeats=128;
        for(int i=0;i<16;++i){
          sources[i]=reinterpret_cast<uint64_t>(y.GetPhyAddr())+((row+i)*width+base*16)*2;
          destinations[i]=reinterpret_cast<uint64_t>(tmp.GetPhyAddr())+(row*width+base*256+i*16)*2;
        }
        TransDataTo5HD<half>(destinations,sources,TransDataTo5HDParams{false,false,
          static_cast<uint8_t>(repeats),static_cast<uint16_t>(repeats==1?0:16),
          static_cast<uint16_t>(repeats==1?0:1)});
        pipe_barrier(PIPE_V);
      }
      for(int64_t col=0;col<width/16;col+=128){
        int64_t repeats=width/16-col;if(repeats>128)repeats=128;
        for(int i=0;i<16;++i){
          sources[i]=reinterpret_cast<uint64_t>(tmp.GetPhyAddr())+(row*width+col*256+i*16)*2;
          destinations[i]=reinterpret_cast<uint64_t>(packed.GetPhyAddr())+(col*tile*16+(row+i)*16)*2;
        }
        TransDataTo5HD<half>(destinations,sources,TransDataTo5HDParams{false,false,
          static_cast<uint8_t>(repeats),static_cast<uint16_t>(repeats==1?0:tile),
          static_cast<uint16_t>(repeats==1?0:16)});
        pipe_barrier(PIPE_V);
      }
      }
      pipe_barrier(PIPE_V);
#ifdef C8_NZ_COMPACT_SCALE
      if(valid){
        for(int64_t h=0;h<heads;++h){
          Gather(compact,scale,indices,static_cast<uint32_t>(h*2),static_cast<uint32_t>(tile));
          pipe_barrier(PIPE_V);
          Brcb(broadcast,compact,static_cast<uint8_t>(tile/8),BrcbRepeatParams{1,8});
          pipe_barrier(PIPE_V);
          for(int64_t col=h*dim/16;col<(h+1)*dim/16;++col){
            Mul(packed[col*tile*16],packed[col*tile*16],broadcast,tile*16);
            pipe_barrier(PIPE_V);
          }
        }
      }
      pipe_barrier(PIPE_V);
#endif
      int64_t dst=(block*(width/16)*block_size+offset)*16;
#ifdef C8_NZ_PIPELINE
      SetFlag<HardEvent::V_MTE3>(outputReady);WaitFlag<HardEvent::V_MTE3>(outputReady);
#else
      pipe_barrier(PIPE_ALL);
#endif
      DataCopy(outputs[side][dst],packed,DataCopyParams{static_cast<uint16_t>(width/16),
        static_cast<uint16_t>(tile),0,static_cast<uint16_t>(block_size-tile)});
#ifdef C8_NZ_PIPELINE
      if(buffer==0)SetFlag<HardEvent::MTE3_V>(outputDone0);
      else SetFlag<HardEvent::MTE3_V>(outputDone1);
      pending[buffer]=true;buffer=1-buffer;
      // The single input/scale buffers may be overwritten only after vector reads finish.
      SetFlag<HardEvent::V_MTE2>(inputConsumed);WaitFlag<HardEvent::V_MTE2>(inputConsumed);
#else
      pipe_barrier(PIPE_ALL);
#endif
    }
  }
#ifdef C8_NZ_PIPELINE
  if(pending[0])WaitFlag<HardEvent::MTE3_V>(outputDone0);
  if(pending[1])WaitFlag<HardEvent::MTE3_V>(outputDone1);
  pipe_barrier(PIPE_ALL);
#endif
}
