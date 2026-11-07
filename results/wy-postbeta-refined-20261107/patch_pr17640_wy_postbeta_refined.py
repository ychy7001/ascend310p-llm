# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: use existing refined solver on every K/V=128 task."""
import hashlib
from patch_pr17640_wy_postbeta_neumaier import patch_text as parent_patch
PARENT_SHA='e53600af2172159338673fc15c153fd5b40224265048ab41108af9de4804503e'
OLD='''    if (allowCompDoubling && useFp32ForwardSubstitution && kHeadDim_ == 128 && vHeadDim_ == 128) {
      return ProcessRefinedTask(b, kHeadIdx, vHeadIdx, tokenStart);
    }'''
NEW='''    // Isolated diagnostic: force the existing refined path for all 128-wide
    // tasks. Retain the original gate constants and scalar retry behavior.
    if (allowCompDoubling && kHeadDim_ == 128 && vHeadDim_ == 128) {
      return ProcessRefinedTask(b, kHeadIdx, vHeadIdx, tokenStart);
    }'''
def patch_text(data):
 parent=parent_patch(data)
 if hashlib.sha256(parent).hexdigest()!=PARENT_SHA:raise ValueError('unexpected Neumaier parent')
 text=parent.decode()
 if text.count(OLD)!=1:raise ValueError('expected one refined dispatch')
 return text.replace(OLD,NEW).encode()
