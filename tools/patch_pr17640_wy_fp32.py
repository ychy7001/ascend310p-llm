# SPDX-License-Identifier: Apache-2.0
"""Patch only pinned PR17640 WY precision; no engine/gating edits or fallback."""
import argparse
import hashlib
from pathlib import Path

KERNEL = Path('csrc/moe/chunk_gated_delta_rule_compute_wy/op_kernel/arch20/compute_wy_kernel.h')
UPSTREAM_GIT_BLOB = '6b2f061fc8d3440bf0942a36c17196181d1d3f8e'

GRAM_ANCHOR = '''    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Muls(attnLocal, attnLocal, -1.0f, ATTEN_ELEMS);'''

GRAM_REPLACEMENT = '''    // beta*K is FP32 in the WY equation. Casting it to a single half
    // before the Gram product perturbs the triangular system, especially
    // with correlated keys and weak decay. K is already exactly half;
    // retain the beta*K rounding residual as a second cube operand.
    for (uint32_t k0 = 0; k0 < kHeadDim_; k0 += FIXED_CHUNK_SIZE) {
      const uint32_t width = (kHeadDim_ - k0) < FIXED_CHUNK_SIZE ? (kHeadDim_ - k0) : FIXED_CHUNK_SIZE;
      Cast(scratch, qHalf[k0], RoundMode::CAST_NONE, static_cast<uint64_t>(width), FIXED_CHUNK_SIZE,
           {1, 1, 8, static_cast<uint8_t>(alignK_ * sizeof(half) / BLOCK_BYTES)});
      PipeBarrier<PIPE_V>();
      Sub(scratch, rhs[k0], scratch, static_cast<uint64_t>(width), FIXED_CHUNK_SIZE,
          {1, 1, 1, 8, static_cast<uint8_t>(alignK_ / FLOAT_PER_BLOCK), 8});
      PipeBarrier<PIPE_V>();
      Cast(qHalf[k0], scratch, RoundMode::CAST_NONE, static_cast<uint64_t>(width), FIXED_CHUNK_SIZE,
           {1, 1, static_cast<uint8_t>(alignK_ * sizeof(half) / BLOCK_BYTES), 8});
      PipeBarrier<PIPE_V>();
    }
    microMm_.GramAcc(scratch, qHalf, halfLocal, storeBuf_.Get<half>().ReinterpretCast<float>(),
                     kHeadDim_, alignK_, alignK_);
    Add(attnLocal, attnLocal, scratch, ATTEN_ELEMS);
    PipeBarrier<PIPE_V>();
    SyncEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Muls(attnLocal, attnLocal, -1.0f, ATTEN_ELEMS);'''

SCAN_OLD = '''    float cumulative = 0.0f;
    for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
      cumulative += gRaw.GetValue(i);
      gLocal.SetValue(i, cumulative);
    }'''

SCAN_NEW = '''    // Keep prefix-sum rounding from accumulating across the chunk. The
    // gate output is FP32 and feeds both the Gram decay and the H/O state.
    float cumulative = 0.0f;
    float compensation = 0.0f;
    for (uint32_t i = 0; i < FIXED_CHUNK_SIZE; ++i) {
      const float corrected = gRaw.GetValue(i) - compensation;
      const float next = cumulative + corrected;
      compensation = (next - cumulative) - corrected;
      cumulative = next;
      gLocal.SetValue(i, cumulative);
    }'''


def patch_text(data):
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != UPSTREAM_GIT_BLOB:
        raise ValueError('Expected unmodified PR17640 18163f6b WY kernel')
    text = data.decode()
    assert text.count(GRAM_ANCHOR) == text.count(SCAN_OLD) == 1
    text = text.replace(GRAM_ANCHOR, GRAM_REPLACEMENT).replace(SCAN_OLD, SCAN_NEW)
    start = text.index('// Additional empirical admission for distributed, small coefficients.')
    end = text.index('__aicore__ inline uint32_t AlignUp', start)
    text = text[:start] + text[end:]
    start = text.index('  __aicore__ inline float FrobeniusSquared(')
    end = text.index('  // Exact in-place solve', start)
    text = text[:start] + text[end:]
    start = text.index('    if (allowCompDoubling && useFp32ForwardSubstitution &&\n')
    end = text.index('    if (allowCompDoubling && useFp32ForwardSubstitution && kHeadDim_ == 128', start)
    text = text[:start] + text[end:]
    return text.encode()


def apply(source_root):
    target = Path(source_root) / KERNEL
    patched = patch_text(target.read_bytes())
    target.write_bytes(patched)
    return hashlib.sha256(patched).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    print(apply(args.source_root))


if __name__ == '__main__':
    main()
