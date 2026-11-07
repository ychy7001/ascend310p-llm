# SPDX-License-Identifier: Apache-2.0
"""Isolated scan change on top of the frozen raw-K Gram postbeta candidate."""
import hashlib
from patch_pr17640_wy_gram_postbeta128 import patch_text as gram_patch
PARENT_SHA='5a0d155954bfb5c9965cddbaca87a79ee094ee389ee1152c6afea406a193c40d'
OLD='''    float cumulative = 0.0f;
    for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
      cumulative += gRaw.GetValue(i);
      gLocal.SetValue(i, cumulative);
    }
'''
NEW='''    float cumulative = 0.0f;
    float correction = 0.0f;
    for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
      const float value = gRaw.GetValue(i);
      const float next = cumulative + value;
      const float absSum = cumulative < 0.0f ? -cumulative : cumulative;
      const float absValue = value < 0.0f ? -value : value;
      if (absSum >= absValue) {
        correction += (cumulative - next) + value;
      } else {
        correction += (value - next) + cumulative;
      }
      cumulative = next;
      gLocal.SetValue(i, cumulative + correction);
    }
'''
def patch_text(data):
 parent=gram_patch(data)
 if hashlib.sha256(parent).hexdigest()!=PARENT_SHA:raise ValueError('unexpected parent candidate')
 text=parent.decode()
 if text.count(OLD)!=1:raise ValueError('expected exactly one scalar scan')
 return text.replace(OLD,NEW).encode()
