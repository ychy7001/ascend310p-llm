# SPDX-License-Identifier: Apache-2.0
"""在隔离目录生成MindIEQ grouped实验头；保留外部版权，不改原文件。"""
import argparse
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--native-layout',action='store_true',help='读取现有native signed W4布局，片上转biased，不复制完整权重')
a=p.parse_args();s=a.source.read_text()
s=s.replace('    __aicore__ FORCE_INLINE void Process()','    __aicore__ FORCE_INLINE void Process(uint32_t first_block)',1)
needle='for (uint32_t loop_idx = block_idx; loop_idx < core_loop_; loop_idx += block_num)'
assert s.count(needle)==1;s=s.replace(needle,'for (uint32_t loop_idx = first_block; loop_idx < core_loop_; loop_idx += block_num)')
needle='AscendC::Nz2NdParamsFull(1, m_round, n_round, 0, m_round, n_, 0)'
assert s.count(needle)==1
s=s.replace(needle,'AscendC::Nz2NdParamsFull(1, m_actual, n_actual, 0, m_round, n_, 0)')
# ND int8的单行/尾单行仍按32元素K分形加载；不复用原V220单行分支。
start=s.index('        if (((m_ == 1) || (m_actual == 1)) && FormatA == CubeFormat::ND) {')
end=s.index('        } else {',start)+len('        } else {')
s=s[:start]+'        {'+s[end:]
if a.native_layout:
    # tile staging直接从native [N16,K64,16,32]读取，避免全局布局转换。
    start=s.index('        AscendC::DataCopy(\n            ub_tile_staging, gm_b[gmByteOffset]')
    end=s.index('        // MTE2→V 用 flag',start)
    s=s[:start]+'''        const uint64_t bankBytes = (uint64_t)nOrgUp * (kOrgUp >> 1);
        const uint64_t bankBase = (gmByteOffset / bankBytes) * bankBytes;
        const uint64_t within = gmByteOffset % bankBytes;
        const uint64_t firstK = within / ((uint64_t)nOrgUp * 32);
        const uint64_t firstN = (within % ((uint64_t)nOrgUp * 32)) / 32;
        const uint64_t nativeBase = bankBase + (firstN / 16) * (kOrgUp / 64) * 512 + firstK * 512;
        for (uint32_t g=0; g<kGrpLoop; ++g) {
            AscendC::DataCopy(ub_tile_staging[g*nRound*32], gm_b[nativeBase+g*512],
                AscendC::DataCopyParams(nRound/16,16,(kOrgUp/64-1)*16,0));
        }
        SET_FLAG(MTE2,V,EVENT_ID3);
        const uint32_t seg1Grps=0;
'''+s[end:]
    # XOR0x88 = v+136-2*(v & 136)；INT16位运算，signed code转biased。
    needle='        // 3) biased 展开：'
    pos=s.index(needle)
    s=s[:pos]+'''        auto maskbuf = buf.template GetBuffer<BufferType::ASCEND_UB,int16_t>(UB_B_T2_OFF_INT4);
        auto xorconst = buf.template GetBuffer<BufferType::ASCEND_UB,int16_t>(UB_B_CONST_OFF_INT4+256);
        AscendC::Duplicate<int16_t>(xorconst,(int16_t)136,128);
        SetVectorMask<int8_t>((uint64_t)-1,(uint64_t)-1);
        AscendC::And<int16_t,false>(maskbuf,ub_v16,xorconst,(uint64_t)0,repeats,
            AscendC::BinaryRepeatParams(1,1,0,8,8,0));
        AscendC::Muls<int16_t>(maskbuf,maskbuf,(int16_t)2,byteCount);
        AscendC::Sub<int16_t>(ub_v16,ub_v16,maskbuf,byteCount);
        AscendC::Adds<int16_t>(ub_v16,ub_v16,(int16_t)136,byteCount);
'''+s[pos:]
# 相对依赖仍引用原构建后的目录，仅修改实验类本体。
base=a.source.parent
import re
s=re.sub(r'#include "([^"]+)"',lambda m:'#include "'+str((base/m[1]).resolve())+'"',s)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:f.write('// Experimental modifications: grouped scheduling and exact ND tail output.\n'+s)
