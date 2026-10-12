// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

// One core owns one KV head and up to 16 GQA query heads. K/V tiles are
// shared across query heads; dequantization and softmax state stay in UB.
extern "C" __global__ __aicore__ void c8_attention_cube_kernel(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR ks, GM_ADDR vs,
    GM_ADDR table, GM_ADDR output, int64_t queries, int64_t query_heads,
    int64_t kv_heads, int64_t dim, int64_t block_size, int64_t num_blocks,
    int64_t sequence_length, int64_t splits) {
  int64_t id=GetBlockIdx(), groups=query_heads/kv_heads;
  if (id>=queries*kv_heads*splits) return;
  int64_t part=id/(queries*kv_heads), owner=id%(queries*kv_heads);
  int64_t qi=owner/kv_heads, head=owner%kv_heads;
  int64_t visible=sequence_length-queries+qi+1;
  int64_t chunk=((sequence_length+splits*16-1)/(splits*16))*16;
  int64_t begin=part*chunk, end=begin+chunk;
  if(end>visible)end=visible;
  TPipe pipe;
  TBuf<TPosition::VECCALC> qb, ib, khb, vhb, fbuf, sb, scoreb, probb, accb, resultb, redb, tmpb;
  TBuf<TPosition::A1> a1b; TBuf<TPosition::B1> b1b;
  TBuf<TPosition::A2> a2b; TBuf<TPosition::B2> b2b; TBuf<TPosition::CO1> c1b;
  pipe.InitBuffer(qb,16*256*2); pipe.InitBuffer(ib,16*256);
  pipe.InitBuffer(khb,16*256*2); pipe.InitBuffer(vhb,16*256*2);
  pipe.InitBuffer(fbuf,16*256*4); pipe.InitBuffer(sb,16*16*2);
  pipe.InitBuffer(scoreb,256*4); pipe.InitBuffer(probb,256*2);
  pipe.InitBuffer(accb,16*256*4); pipe.InitBuffer(resultb,256*4);
  pipe.InitBuffer(redb,32*4); pipe.InitBuffer(tmpb,256*4);
  pipe.InitBuffer(a1b,16*256*2); pipe.InitBuffer(b1b,16*256*2);
  pipe.InitBuffer(a2b,16*256*2); pipe.InitBuffer(b2b,16*256*2);
  pipe.InitBuffer(c1b,256*4);
  auto q=qb.Get<half>(); auto xi=ib.Get<int8_t>();
  auto kh=khb.Get<half>(); auto vh=vhb.Get<half>(); auto xf=fbuf.Get<float>();
  auto scales=sb.Get<half>(); auto scores=scoreb.Get<float>(); auto prob=probb.Get<half>();
  auto acc=accb.Get<float>(); auto result=resultb.Get<float>();
  auto red=redb.Get<float>(); auto scratch=tmpb.Get<float>();
  auto a1=a1b.Get<half>(); auto b1=b1b.Get<half>();
  auto a2=a2b.Get<half>(); auto b2=b2b.Get<half>(); auto c1=c1b.Get<float>();
  GlobalTensor<half> qgm,ksgm,vsgm; GlobalTensor<int8_t> kgm,vgm;
  GlobalTensor<int32_t> mapping; GlobalTensor<float> dst;
  qgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(query));
  kgm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
  vgm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(value));
  ksgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ks));
  vsgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vs));
  mapping.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
  Duplicate(q,static_cast<half>(0),16*dim);
  Duplicate(acc,0.0f,16*dim); pipe_barrier(PIPE_ALL);
  DataCopy(q,qgm[(qi*query_heads+head*groups)*dim],groups*dim);
  pipe_barrier(PIPE_ALL);
  float maximum[16], denominator[16];
  for(int r=0;r<16;++r){maximum[r]=-1.0e30f;denominator[r]=0.0f;}
  DataCopyParams pack{16,1,static_cast<uint16_t>(dim/16-1),0};
  for(int64_t j=0;j<dim/16;++j) DataCopy(a1[j*256],q[j*16],pack);
  pipe_barrier(PIPE_ALL);
  LoadData2DParams load; load.startIndex=0;load.repeatTimes=dim/16;
  load.srcStride=1;load.dstGap=0;load.ifTranspose=false;
  LoadData(a2,a1,load); pipe_barrier(PIPE_ALL);
  // Save Q in L1; A2 will be reused by PV and reloaded for every QK tile.
  for(int64_t base=begin;base<end;base+=16){
    int64_t physical=mapping.GetValue(base/block_size);
    if(physical<0||physical>=num_blocks)continue;
    int64_t count=end-base;if(count>16)count=16;
    int64_t slot=physical*block_size+base%block_size;
    Duplicate(kh,static_cast<half>(0),16*dim);
    Duplicate(vh,static_cast<half>(0),16*dim);pipe_barrier(PIPE_ALL);
    DataCopyParams dma{static_cast<uint16_t>(count),static_cast<uint16_t>(dim/32),
                       static_cast<uint16_t>((kv_heads-1)*dim/32),0};
    DataCopy(xi,kgm[(slot*kv_heads+head)*dim],dma);
    DataCopy(scales,ksgm[slot*16],count*16);pipe_barrier(PIPE_ALL);
    Cast(kh,xi,RoundMode::CAST_NONE,count*dim);pipe_barrier(PIPE_V);
    Cast(xf,kh,RoundMode::CAST_NONE,count*dim);pipe_barrier(PIPE_ALL);
    for(int64_t t=0;t<count;++t)Muls(xf[t*dim],xf[t*dim],static_cast<float>(scales.GetValue(t*16+head)),dim);
    pipe_barrier(PIPE_V);Cast(kh,xf,RoundMode::CAST_NONE,count*dim);pipe_barrier(PIPE_ALL);
    for(int64_t j=0;j<dim/16;++j)DataCopy(b1[j*256],kh[j*16],pack);
    pipe_barrier(PIPE_ALL);
    load.repeatTimes=dim/16;load.ifTranspose=false;
    // A1 currently holds PV from the prior tile; repack Q.
    for(int64_t j=0;j<dim/16;++j)DataCopy(a1[j*256],q[j*16],pack);
    pipe_barrier(PIPE_ALL);LoadData(a2,a1,load);LoadData(b2,b1,load);pipe_barrier(PIPE_ALL);
    MmadParams mm;mm.m=16;mm.n=16;mm.k=dim;mm.cmatrixInitVal=true;
    Mmad(c1,a2,b2,mm);pipe_barrier(PIPE_ALL);
    DataCopyEnhancedParams enhanced;enhanced.blockMode=BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(scores,c1,DataCopyParams{1,1,0,0},enhanced);pipe_barrier(PIPE_ALL);
    float factor=dim==256?0.0625f:0.08838834764831844f;
    Duplicate(prob,static_cast<half>(0),256);pipe_barrier(PIPE_ALL);
    for(int64_t r=0;r<groups;++r){
      Muls(scores[r*16],scores[r*16],factor,16);pipe_barrier(PIPE_ALL);
      float next=maximum[r];
      for(int64_t t=0;t<count;++t){float s=scores.GetValue(r*16+t);if(s>next)next=s;}
      Duplicate(red,maximum[r]-next,8);pipe_barrier(PIPE_V);Exp(red,red,8);pipe_barrier(PIPE_ALL);
      float alpha=denominator[r]==0.0f?0.0f:red.GetValue(0);
      Muls(acc[r*dim],acc[r*dim],alpha,dim);
      for(int64_t t=count;t<16;++t)scores.SetValue(r*16+t,-1.0e30f);
      pipe_barrier(PIPE_ALL);Adds(scores[r*16],scores[r*16],-next,16);pipe_barrier(PIPE_V);
      Exp(scores[r*16],scores[r*16],16);pipe_barrier(PIPE_V);
      ReduceSum(red,scores[r*16],scratch,16);pipe_barrier(PIPE_ALL);
      denominator[r]=denominator[r]*alpha+red.GetValue(0);maximum[r]=next;
    }
    pipe_barrier(PIPE_ALL);
    for(int64_t r=groups;r<16;++r)Duplicate(scores[r*16],0.0f,16);
    pipe_barrier(PIPE_V);
    Cast(prob,scores,RoundMode::CAST_NONE,256);
    pipe_barrier(PIPE_ALL);
    DataCopy(xi,vgm[(slot*kv_heads+head)*dim],dma);
    DataCopy(scales,vsgm[slot*16],count*16);pipe_barrier(PIPE_ALL);
    Cast(vh,xi,RoundMode::CAST_NONE,count*dim);pipe_barrier(PIPE_V);
    Cast(xf,vh,RoundMode::CAST_NONE,count*dim);pipe_barrier(PIPE_ALL);
    for(int64_t t=0;t<count;++t)Muls(xf[t*dim],xf[t*dim],static_cast<float>(scales.GetValue(t*16+head)),dim);
    pipe_barrier(PIPE_V);Cast(vh,xf,RoundMode::CAST_NONE,count*dim);pipe_barrier(PIPE_ALL);
    DataCopy(a1,prob,256);pipe_barrier(PIPE_ALL);
    load.repeatTimes=1;load.ifTranspose=false;LoadData(a2,a1,load);
    for(int64_t j=0;j<dim/16;++j){
      DataCopy(b1,vh[j*16],pack);pipe_barrier(PIPE_ALL);
      load.ifTranspose=true;LoadData(b2,b1,load);pipe_barrier(PIPE_ALL);
      mm.k=16;Mmad(c1,a2,b2,mm);pipe_barrier(PIPE_ALL);
      DataCopy(result,c1,DataCopyParams{1,1,0,0},enhanced);pipe_barrier(PIPE_ALL);
      for(int64_t r=0;r<groups;++r)Add(acc[r*dim+j*16],acc[r*dim+j*16],result[r*16],16);
      pipe_barrier(PIPE_ALL);
    }
  }
  if(splits==1){
    for(int64_t r=0;r<groups;++r){
      if(denominator[r]>0.0f)Muls(acc[r*dim],acc[r*dim],1.0f/denominator[r],dim);
    }
    pipe_barrier(PIPE_ALL);
    DataCopy(dst[(qi*query_heads+head*groups)*dim],acc,groups*dim);
  }else{
    for(int64_t r=0;r<groups;++r){
      int64_t offset=((part*queries+qi)*query_heads+head*groups+r)*(dim+32);
      Duplicate(red,0.0f,32);pipe_barrier(PIPE_ALL);
      red.SetValue(0,maximum[r]);red.SetValue(1,denominator[r]);
      pipe_barrier(PIPE_ALL);
      DataCopy(dst[offset],acc[r*dim],dim);
      DataCopy(dst[offset+dim],red,32);pipe_barrier(PIPE_ALL);
    }
  }
  pipe_barrier(PIPE_ALL);
}

// Stable online-softmax merge of independently computed KV ranges.
extern "C" __global__ __aicore__ void c8_attention_merge_kernel(
    GM_ADDR partial, GM_ADDR output, int64_t rows, int64_t dim, int64_t splits){
  int64_t row=GetBlockIdx();if(row>=rows)return;
  TPipe pipe;TBuf<TPosition::VECCALC> xb,ob,sb,eb;
  pipe.InitBuffer(xb,256*4);pipe.InitBuffer(ob,256*4);
  pipe.InitBuffer(sb,32*4);pipe.InitBuffer(eb,8*4);
  auto x=xb.Get<float>();auto o=ob.Get<float>();
  auto state=sb.Get<float>();auto e=eb.Get<float>();
  GlobalTensor<float> src,dst;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
  Duplicate(o,0.0f,dim);pipe_barrier(PIPE_ALL);
  float maximum=-1.0e30f,denominator=0.0f;
  for(int64_t part=0;part<splits;++part){
    int64_t offset=(part*rows+row)*(dim+32);
    DataCopy(state,src[offset+dim],32);pipe_barrier(PIPE_ALL);
    float m=state.GetValue(0),l=state.GetValue(1);
    if(l<=0.0f)continue;
    float next=maximum>m?maximum:m;
    Duplicate(e,maximum-next,8);pipe_barrier(PIPE_V);Exp(e,e,8);pipe_barrier(PIPE_ALL);
    float alpha=denominator==0.0f?0.0f:e.GetValue(0);
    Duplicate(e,m-next,8);pipe_barrier(PIPE_V);Exp(e,e,8);pipe_barrier(PIPE_ALL);
    float beta=e.GetValue(0);
    DataCopy(x,src[offset],dim);pipe_barrier(PIPE_ALL);
    Muls(o,o,alpha,dim);Muls(x,x,beta,dim);pipe_barrier(PIPE_V);
    Add(o,o,x,dim);pipe_barrier(PIPE_ALL);
    denominator=denominator*alpha+l*beta;maximum=next;
  }
  if(denominator>0.0f)Muls(o,o,1.0f/denominator,dim);
  pipe_barrier(PIPE_ALL);DataCopy(dst[row*dim],o,dim);pipe_barrier(PIPE_ALL);
}
