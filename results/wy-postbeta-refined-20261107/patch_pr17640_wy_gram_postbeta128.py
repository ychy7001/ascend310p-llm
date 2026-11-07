# SPDX-License-Identifier: Apache-2.0
"""Isolated K/V=128 diagnostic: raw-K library Gram followed by FP32 beta.

Avoid rounding beta*K to half before Gram. Preserve RHS construction, gates,
solve/apply, scan, output storage, small dimensions and production bindings.
"""
import hashlib
from pathlib import Path
KERNEL=Path('csrc/moe/chunk_gated_delta_rule_compute_wy/op_kernel/arch20/compute_wy_kernel.h')
UPSTREAM_GIT_BLOB='6b2f061fc8d3440bf0942a36c17196181d1d3f8e'
OLD='''    if (kHeadDim_ >= 128) {
      microMm_.GramAcc(attnLocal, qHalf, halfLocal, storeBuf_.Get<half>().ReinterpretCast<float>(), kHeadDim_,
                       kHeadDim_, alignK_);
    } else {
'''
NEW='''    if (kHeadDim_ >= 128) {
      // Both Cube operands now contain the original FP16 K. Apply beta to
      // FP32 Gram rows instead of rounding beta*K to FP16 before the dot.
      Muls(qHalf, halfLocal, static_cast<half>(1), chunkKElems_);
      PipeBarrier<PIPE_V>();
      cubeGemm_.GemmATransB(attnLocal, qHalf, halfLocal, scratch,
                            storeBuf_.Get<half>(), kHeadDim_, kHeadDim_, alignK_);
      BroadcastMulRowsFloat(attnLocal, attnLocal, betaLocal, scratch,
                            FIXED_CHUNK_SIZE, FIXED_CHUNK_SIZE,
                            FIXED_CHUNK_SIZE, FIXED_CHUNK_SIZE);
    } else {
'''
def patch_text(data):
 blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 if blob!=UPSTREAM_GIT_BLOB:raise ValueError('expected unmodified pinned PR17640 kernel')
 text=data.decode()
 if text.count(OLD)!=1:raise ValueError('expected exactly one Gram dispatch')
 return text.replace(OLD,NEW).encode()
