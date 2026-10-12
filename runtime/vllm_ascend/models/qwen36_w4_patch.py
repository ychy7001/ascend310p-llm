# SPDX-License-Identifier: Apache-2.0
"""Opt-in Qwen3.6 packed-checkpoint -> 310P W4A8 model adapter.

This is loaded only in the isolated experiment fork.  The stock Qwen3.5
decoder, GDN/attention, and non-expert weight loader remain in charge of all
other tensors; this patch replaces only the routed-MoE block when the model
config carries ``ascend_expert_quantization`` metadata.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import torch

from vllm.model_executor.models import qwen3_5

try:
    from vllm.distributed import get_tp_group
except Exception as exc:  # pragma: no cover - a clear opt-in failure is preferable
    raise RuntimeError("Qwen3.6 W4 patch requires the isolated OpenSensor W4 modules") from exc

ASCEND_QWEN4EXP_DTYPE_POLICY = None
W4SparseMoE = None


def _w4_types():
    global ASCEND_QWEN4EXP_DTYPE_POLICY, W4SparseMoE
    if W4SparseMoE is None:
        from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY as policy
        from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE as moe
        ASCEND_QWEN4EXP_DTYPE_POLICY, W4SparseMoE = policy, moe
    return ASCEND_QWEN4EXP_DTYPE_POLICY, W4SparseMoE


_SIDE_CAR = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_offset)$"
)
_ROUTER = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
_SHARED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight$"
)
_SHARED_GATE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.shared_expert_gate\.weight$")
_PACKED_EXPERT = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$")


def _metadata(config: object) -> dict | None:
    value = getattr(config, "ascend_expert_quantization", None)
    if not isinstance(value, dict) or value.get("format") != "qwen36_modelslim_ssz_w4a8_v1":
        return None
    return value


def _text_config(obj: object) -> object:
    """Return the HF text config for both the causal and multimodal roots."""
    config = getattr(obj, "config", None)
    if config is None:
        return None
    return getattr(config, "text_config", None) or getattr(config, "hf_text_config", None) or config


def _layer(self, index: int):
    root = getattr(self, "language_model", self)
    return root.model.layers[index].mlp


_decoder_init = qwen3_5.Qwen3_5DecoderLayer.__init__


def _decoder_init_w4(self, vllm_config, layer_type: str, prefix: str = ""):
    config = vllm_config.model_config.hf_text_config
    use_w4 = _metadata(config) is not None and getattr(config, "model_type", "") in (
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    )
    # MTP uses the original BF16 experts, with no W4 sidecar.  Its decoder
    # shares the target config, so metadata alone must not select W4 here.
    if not use_w4 or "mtp" in prefix.split("."):
        _decoder_init(self, vllm_config, layer_type, prefix)
        return
    policy, moe_cls = _w4_types()
    parallel = vllm_config.parallel_config
    if getattr(parallel, "enable_eplb", False) or getattr(parallel, "use_sequence_parallel_moe", False):
        raise ValueError("experimental Qwen3.6 W4 bank requires static ownership without EPLB or sequence parallel MoE")

    # The stock decoder constructs the full BF16 FusedMoE before returning.
    # On a 310P that allocation can OOM before the W4 replacement runs. Keep
    # the stock attention/norm construction, but make its MoE branch empty.
    original_moe = qwen3_5.Qwen3NextSparseMoeBlock

    class _MoePlaceholder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    qwen3_5.Qwen3NextSparseMoeBlock = _MoePlaceholder
    try:
        _decoder_init(self, vllm_config, layer_type, prefix)
    finally:
        qwen3_5.Qwen3NextSparseMoeBlock = original_moe
    ep = get_tp_group()
    self.mlp = moe_cls(
        config=config,
        dtype_policy=policy,
        expert_sharding=(getattr(ep, "rank_in_group", 0), getattr(ep, "world_size", 1)),
    )


def _load_special(self, name: str, tensor: torch.Tensor) -> bool:
    match = _SIDE_CAR.fullmatch(name)
    if match:
        layer, expert, projection, kind = match.groups()
        _layer(self, int(layer)).load_projection(int(expert), projection, kind, tensor)
        # Peer-owned tensors must also be consumed, never passed to FusedMoE.
        return True
    match = _ROUTER.fullmatch(name)
    if match:
        target = _layer(self, int(match.group(1))).gate
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
        return True
    match = _SHARED.fullmatch(name)
    if match:
        moe = _layer(self, int(match.group(1)))
        projection = match.group(2)
        target = {
            "gate_proj": moe.shared_gate_up[: moe.local_shared_inter],
            "up_proj": moe.shared_gate_up[moe.local_shared_inter :],
            "down_proj": moe.shared_down,
        }[projection]
        if not moe.shared_expert_replicated:
            start = moe.expert_tp_rank * moe.local_shared_inter
            tensor = tensor[:, start : start + moe.local_shared_inter] if projection == "down_proj" else tensor[start : start + moe.local_shared_inter]
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
        return True
    match = _SHARED_GATE.fullmatch(name)
    if match:
        target = _layer(self, int(match.group(1))).shared_expert_gate
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
        return True
    return False


def _load_weights_w4(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    config = _text_config(self)
    if _metadata(config) is None or getattr(self, "_qwen36_parent_loader_active", False):
        return _original_loader(self)(self, weights)
    handled: set[str] = set()

    def remaining():
        for name, tensor in weights:
            if _load_special(self, name, tensor):
                handled.add(name)
                continue
            # The original BF16 packed expert payload is deliberately absent
            # from the merged W4 index; keep this guard for stale indexes.
            if _PACKED_EXPERT.fullmatch(name):
                handled.add(name)
                continue
            yield name, tensor

    root = getattr(self, "language_model", self)
    if root is not self:
        root._qwen36_parent_loader_active = True
    try:
        loaded = _original_loader(self)(self, remaining())
    finally:
        if root is not self:
            del root._qwen36_parent_loader_active
    # vLLM checks destination parameter names, not checkpoint tensor names.
    # Verify every local expert slice before marking its packed bank loaded.
    destinations = set()
    root = getattr(self, "language_model", self)
    destination_prefix = "language_model." if root is not self else ""
    for index, layer in enumerate(root.model.layers):
        moe = layer.mlp
        if not isinstance(moe, _w4_types()[1]):
            continue
        source_prefix = f"model.language_model.layers.{index}.mlp."
        required = {source_prefix + "gate.weight"}
        if moe.has_shared_expert:
            required |= {
                source_prefix + f"shared_expert.{projection}.weight"
                for projection in ("gate_proj", "up_proj", "down_proj")
            }
            required.add(source_prefix + "shared_expert_gate.weight")
        required |= {
            source_prefix + f"experts.{expert}.{projection}.{kind}"
            for expert in range(moe.expert_offset, moe.expert_offset + moe.num_local_experts)
            for projection in ("gate_proj", "up_proj", "down_proj")
            for kind in ("weight", "weight_scale", "weight_offset")
        }
        missing = required - handled
        if missing:
            raise ValueError(f"W4 layer {index} missing {len(missing)} checkpoint tensors: {sorted(missing)[:5]}")
        prefix = destination_prefix + f"model.layers.{index}.mlp."
        destinations |= {prefix + name for name, _ in moe.named_parameters()}
    return set(loaded) | destinations


_load_weights_originals = {
    qwen3_5.Qwen3_5ForConditionalGeneration: qwen3_5.Qwen3_5ForConditionalGeneration.load_weights,
    qwen3_5.Qwen3_5MoeForCausalLM: qwen3_5.Qwen3_5MoeForCausalLM.load_weights,
}


def _original_loader(self):
    for cls in type(self).__mro__:
        if cls in _load_weights_originals:
            return _load_weights_originals[cls]
    raise TypeError(f"unsupported Qwen3.6 loader root: {type(self).__name__}")


def _set_w4_moe_parameters(self, original):
    if _metadata(_text_config(self)) is None:
        return original(self)
    root = getattr(self, "language_model", self)
    _, moe_cls = _w4_types()
    blocks = [layer.mlp for layer in root.model.layers if isinstance(layer.mlp, moe_cls)]
    if not blocks:
        raise RuntimeError("Qwen3.6 W4 metadata found but no W4 expert blocks constructed")
    example = blocks[0]
    self.expert_weights = []
    self.moe_layers = blocks
    self.num_moe_layers = len(blocks)
    self.num_expert_groups = 1
    self.num_shared_experts = 0
    self.num_logical_experts = example.num_experts
    self.num_physical_experts = example.num_experts
    self.num_local_physical_experts = example.num_local_experts
    self.num_routed_experts = example.num_experts
    self.num_redundant_experts = 0


_causal_moe_parameters = qwen3_5.Qwen3_5MoeForCausalLM.set_moe_parameters
_conditional_moe_parameters = qwen3_5.Qwen3_5MoeForConditionalGeneration.set_moe_parameters
qwen3_5.Qwen3_5MoeForCausalLM.set_moe_parameters = lambda self: _set_w4_moe_parameters(self, _causal_moe_parameters)
qwen3_5.Qwen3_5MoeForConditionalGeneration.set_moe_parameters = lambda self: _set_w4_moe_parameters(self, _conditional_moe_parameters)
qwen3_5.Qwen3_5DecoderLayer.__init__ = _decoder_init_w4
qwen3_5.Qwen3_5ForConditionalGeneration.load_weights = _load_weights_w4
qwen3_5.Qwen3_5MoeForCausalLM.load_weights = _load_weights_w4
