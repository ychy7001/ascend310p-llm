#!/usr/bin/env bash
# Build inside the pinned vLLM-Ascend image after sourcing CANN set_env.sh.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${WORK_DIR:?Set WORK_DIR to a new isolated directory}"
if [[ -e "$WORK_DIR" ]]; then echo 'WORK_DIR must not exist' >&2; exit 2; fi
mkdir -p "$WORK_DIR/artifacts"
git clone --no-checkout https://github.com/opensensor/vllm-ascend.git "$WORK_DIR/community"
git -C "$WORK_DIR/community" checkout --detach 0637d3baa6bc2000ba6433b301f233c6516847b4
git -C "$WORK_DIR/community" submodule update --init csrc/third_party/catlass
cp -a "$project_root/native-overlay/csrc/gmm/." "$WORK_DIR/community/csrc/gmm/"
cd "$WORK_DIR/community/csrc"
export CPATH="$WORK_DIR/community/csrc/third_party/catlass/include${CPATH:+:$CPATH}"
bash build.sh --pkg --ops='qwen_w4_a8_int4_matmul_v310;qwen_w4_a8_pack_v310;qwen_w4_a8_swiglu_pack_v310' --soc=ascend310p
shopt -s nullglob
installers=(build/cann-ops-transformer*.run)
if [[ ${#installers[@]} != 1 ]]; then echo 'Expected one OPP installer' >&2; exit 1; fi
bash "${installers[0]}" --install-path="$WORK_DIR/artifacts/opp"
mkdir -p "$WORK_DIR/artifacts/extension-build"
python3 "$project_root/tools/build_qwen36_int4_extension.py" \
  --community-root "$WORK_DIR/community" \
  --build "$WORK_DIR/artifacts/extension-build"
cp "$WORK_DIR/artifacts/extension-build/qwen36_int4_custom.so" "$WORK_DIR/artifacts/"
echo "EXTENSION=$WORK_DIR/artifacts/qwen36_int4_custom.so"
echo "OPP_DIR=$WORK_DIR/artifacts/opp/vendors/custom_transformer"
