# SPDX-License-Identifier: Apache-2.0
"""在隔离目录生成MindIEQ grouped实验头；保留外部版权，不改原文件。"""
import argparse
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
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
# 相对依赖仍引用原构建后的目录，仅修改实验类本体。
base=a.source.parent
import re
s=re.sub(r'#include "([^"]+)"',lambda m:'#include "'+str((base/m[1]).resolve())+'"',s)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:f.write('// Experimental modifications: grouped scheduling and exact ND tail output.\n'+s)
