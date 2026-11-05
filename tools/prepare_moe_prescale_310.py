#!/usr/bin/env python3
"""Prepare a standalone 310P tiling experiment from an installed CANN source tree.

The caller supplies the frozen QuantGroupedMatmulDequant source directory. No
SDK sources are distributed here. Arithmetic stays in those supplied headers;
the candidate schedules token-scale calculation once before the expert loop.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def prepare(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sorted(source.glob('*.h')):
        shutil.copyfile(path, output / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = output / 'quant_matmul_dequant_grouped.h'
    text = path.read_text()
    old = '__aicore__ inline void Process() {'
    assert text.count(old) == 1
    text = text.replace(old, '__aicore__ inline void Process(bool prescale) {')
    old = '    FirstTiling();\n'
    assert text.count(old) == 1
    text = text.replace(old, old + '''    if (prescale) {
      realM = tilingData->originM;
      ProcessXScale();
      MySyncAllSwift();
    }
''')
    old = '''      if(tilingData->dynamicQuant){
        ProcessXScale();
        MySyncAllSwift();
'''
    assert text.count(old) == 1
    text = text.replace(old, '''      if(prescale) {
        xScaleGm = xScaleGm0[(uint64_t)startM * FLOAT_PERBLOCK];
      } else if(tilingData->dynamicQuant){
        ProcessXScale();
        MySyncAllSwift();
''')
    # GEMV consumes no cross-core data after scale calculation. Its own event
    # dependencies still protect local UB/L1/L0 reuse across adjacent experts.
    assert text.count('if(tilingData->dynamicQuant) MySyncAllSwift();') == 2
    text = text.replace('if(tilingData->dynamicQuant) MySyncAllSwift();',
                        'if(tilingData->dynamicQuant && !prescale) MySyncAllSwift();')
    path.write_text(text)
    fields = sorted(set(re.findall(r'tilingData->(\w+)', '\n'.join(p.read_text() for p in output.glob('*.h')))))
    struct = 'struct QuantMatmulDequantTilingData {\n' + ''.join(f'  uint32_t {f} = 0;\n' for f in fields) + '};\n'
    (output / 'experiment_tiling.h').write_text('#include <cstdint>\n' + struct)
    (output / 'experiment.cpp').write_text('''#include "experiment_tiling.h"
#define ORIG_DTYPE_WEIGHT_SCALE DT_FLOAT
#include "quant_matmul_dequant_grouped.h"
extern "C" __global__ __aicore__ void moe_prescale(
    GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR groups, GM_ADDR y,
    GM_ADDR workspace, uint32_t m, uint32_t k, uint32_t n, uint32_t e,
    uint32_t prescale) {
  QuantMatmulDequantTilingData t;
  t.originM=m; t.originK=k; t.originN=n; t.originE=e;
  t.originKAligned32=k; t.originKAligned512=(k+511)/512*512;
  t.fracK=k/32; t.fracN=n/16; t.CoreNum=8;
  t.dynamicQuant=1; t.perToken=1;
  t.dynamicBaseK=16384; t.dynamicIterK=(k+16383)/16384;
  t.dynamicBaseKTail=(k-1)%16384+1;
  t.singleCoreFracN=(t.fracN+7)/8;
  t.singleCoreFracNTail=(t.fracN-1)%8+1;
  t.baseFracK=16; t.baseFracN=4; t.processXKBaseNMax=1792;
  SetAtomicNone();
  AscendC::QuantMatmulDequantGrouped handle;
  handle.Init(x,w,scale,groups,nullptr,nullptr,nullptr,nullptr,y,workspace,&t);
  handle.Process(prescale != 0);
  SetMaskNorm(); ResetMask();
}
''')
    (output / 'launch.cpp').write_text('''#include <cstdint>
#include "aclrtlaunch_moe_prescale.h"
extern "C" int launch_moe_prescale(void* stream,void* x,void* w,void* scale,
    void* groups,void* y,void* workspace,uint32_t m,uint32_t k,uint32_t n,
    uint32_t e,uint32_t prescale) {
  return ACLRT_LAUNCH_KERNEL(moe_prescale)(8,stream,x,w,scale,groups,y,workspace,m,k,n,e,prescale);
}
''')
    (output / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.20)
project(moe_prescale LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(moe_device STATIC experiment.cpp)
add_library(moe_prescale SHARED launch.cpp)
target_link_libraries(moe_prescale PRIVATE moe_device)
''')
    (output / 'source-manifest.json').write_text(json.dumps({'source_sha256': hashes,
        'change': 'schedule dynamic token scales once before serial expert GEMV loop',
        'arithmetic_headers_unchanged': [p for p in hashes if p != path.name],
        'supported': 'M8/16/24/32; each expert <=8 rows; K256/2048; FP32 weight scales; symmetric W8A8'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
