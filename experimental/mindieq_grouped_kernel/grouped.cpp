// SPDX-License-Identifier: Apache-2.0
// External MindIEQ class retains its own CANN Open Software License V2.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;
using namespace AscendC;
#ifndef DT_INT4
#define DT_INT4 29
#endif
#define ORIG_DTYPE_X2 DT_INT4
#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ 29
#endif
#define FORMAT_X2 FORMAT_FRACTAL_NZ
#include "mindieq_grouped_arch20.h"
extern "C" __global__ __aicore__ void mindieq_grouped_kernel(
 GM_ADDR a,GM_ADDR ascale,GM_ADDR packed,GM_ADDR wscale,GM_ADDR ends,GM_ADDR out,
 int64_t experts,int64_t k,int64_t n,int64_t minimum_rows) {
 GlobalTensor<int64_t> groups;groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
 uint32_t task_prefix=0;
 for(int64_t e=0;e<experts;++e){
  int64_t start=e?groups.GetValue(e-1):0,end=groups.GetValue(e),m=end-start;
  if(m<=0 || m<minimum_rows)continue;
  QuantMatmulPertokenTilingDataArch20 t{};
  t.batchSize=1;t.m=m;t.k=k;t.n=n;t.m0=64;t.n0=128;t.k0=256;
  t.mLoop=(m+63)/64;t.nLoop=(n+127)/128;t.kLoop=(k+255)/256;
  t.coreLoop=t.mLoop*t.nLoop;t.blockDim=GetBlockNum();t.swizzleDirect=0;t.swizzleCount=1;
  t.bL2CacheHint=0;t.withBias=false;t.biasWithBatch=false;
  uint32_t first=(GetBlockIdx()+GetBlockNum()-task_prefix%GetBlockNum())%GetBlockNum();
  PpMatMulNS::QuantBatchMatMulPertokenArch20<0,false,false,CubeFormat::ND,CubeFormat::NZ,CubeFormat::ND> op;
  op.Init(a+start*k,packed+e*n*k/2,nullptr,wscale+e*n*4,ascale+start*4,out+start*n*2,&t);
  op.Process(first);task_prefix+=t.coreLoop;
 }
}
