#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
U=76ea9d24f155afe032ad9ac6c98717f1f7334f36
V=4ba787ec9959e566d3b587a796dff07673324da5
C=41bf90da655bba3c66d0acd7e00abe33960ecfd6
base=quay.io/ascend/cann:9.2.1-310p-ubuntu22.04-py3.12
base_digest=sha256:c4844ab1f074b526c222ea86b6cc55f0f41152c9fc20d6f49a62b259c26dadf2
image=localhost/vllm-ascend:upstream-76ea9d24-310p
mkdir -p source
sha256sum -c source.sha256
if [[ ! -f source/Dockerfile.310p ]]; then
  tar -xzf "main-$U.tar.gz" --strip-components=1 -C source
fi
grep -qx 'ARG VLLM_TAG=v0.33.0' source/Dockerfile.310p
[[ $(git config -f source/.gitmodules --get submodule.csrc/third_party/catlass.commit) == "$C" ]]
catlass=source/csrc/third_party/catlass
if [[ ${2:-} ]]; then
  catlass_repo=$(realpath -- "$2")
  [[ $(git -C "$catlass_repo" rev-parse --verify "$C^{commit}") == "$C" ]]
  mkdir -p "$catlass"
  if [[ ! -e catlass-commit.txt ]]; then
    [[ -z $(find "$catlass" -mindepth 1 -print -quit) ]]
    git -C "$catlass_repo" archive "$C" | tar -x -C "$catlass"
    printf '%s\n' "$C" > catlass-commit.txt
  fi
fi
[[ $(cat catlass-commit.txt) == "$C" && -d $catlass/include ]]
podman image exists "$base@$base_digest" || podman pull --platform linux/amd64 "$base@$base_digest"
[[ $(podman image inspect "$base@$base_digest" --format '{{.Digest}} {{.Os}}/{{.Architecture}}') == "$base_digest linux/amd64" ]]
git_args=()
if [[ ${1:-} ]]; then
  repo=$(realpath -- "$1")
  [[ $(git -C "$repo" rev-parse 'refs/tags/v0.33.0^{commit}') == "$V" ]]
  git -C "$repo" fsck --connectivity-only
  git_args=(-v "$repo:/pinned-vllm.git:ro" --build-arg VLLM_REPO=file:///pinned-vllm.git)
fi
podman build --format docker --jobs=1 --platform linux/amd64 --pull=never --from "$base@$base_digest" --label "org.opencontainers.image.revision=$U" --label "io.vllm.source.revision=$V" "${git_args[@]}" -t "$image" -f source/Dockerfile.310p source
[[ $(podman run --rm --entrypoint git "$image" -C /vllm-workspace/vllm rev-parse HEAD) == "$V" ]]
podman run --rm --entrypoint python3 "$image" -c 'from importlib.metadata import version; print({p: version(p) for p in ("vllm", "vllm-ascend", "torch", "torch-npu")})' > dependencies.txt
podman image inspect "$image" --format '{{.Id}} {{json .Config.Labels}}' > image-id.tmp
mv image-id.tmp image-id.txt
