// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

// Independent two-pass 16-token ND -> raw NZ layout probe; no quantization.
extern "C" __global__ __aicore__ void c8_nz_layout_kernel(
    GM_ADDR input, GM_ADDR output, int64_t width){
  TPipe pipe;TBuf<TPosition::VECCALC> ib,tb,ob;
  pipe.InitBuffer(ib,16*1024*2);pipe.InitBuffer(tb,16*1024*2);pipe.InitBuffer(ob,16*1024*2);
  auto x=ib.Get<half>();auto tmp=tb.Get<half>();auto y=ob.Get<half>();
  GlobalTensor<half> src,dst;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
  DataCopy(x,src,16*width);pipe_barrier(PIPE_ALL);
  uint64_t sources[16],destinations[16];
  for(int i=0;i<16;++i){
    sources[i]=reinterpret_cast<uint64_t>(x.GetPhyAddr())+i*width*2;
    destinations[i]=reinterpret_cast<uint64_t>(tmp.GetPhyAddr())+i*32;
  }
  TransDataTo5HD<half>(destinations,sources,TransDataTo5HDParams{false,false,
    static_cast<uint8_t>(width/16),static_cast<uint16_t>(width==16?0:16),
    static_cast<uint16_t>(width==16?0:1)});
  pipe_barrier(PIPE_V);
  for(int64_t col=0;col<width/16;++col){
    for(int i=0;i<16;++i){
      sources[i]=reinterpret_cast<uint64_t>(tmp.GetPhyAddr())+(col*256+i*16)*2;
      destinations[i]=reinterpret_cast<uint64_t>(y.GetPhyAddr())+(col*256+i*16)*2;
    }
    TransDataTo5HD<half>(destinations,sources,TransDataTo5HDParams{false,false,1,0,0});
    pipe_barrier(PIPE_V);
  }
  pipe_barrier(PIPE_ALL);DataCopy(dst,y,16*width);pipe_barrier(PIPE_ALL);
}
