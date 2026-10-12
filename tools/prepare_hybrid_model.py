#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble an isolated Qwen3.6 BF16-plus-W4-expert model directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open


def link_tree(source: Path, output: Path) -> None:
    for item in source.iterdir():
        if item.name in {"config.json", "model.safetensors.index.json", "manifest.json"} or not item.is_file() or item.name.endswith(".part"):
            continue
        target = output / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(item, target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.source = args.source.resolve()
    args.sidecar = args.sidecar.resolve()
    args.output = args.output.resolve()
    if args.output in (args.source, args.sidecar):
        raise ValueError("output must be separate from source and sidecar")
    config = json.loads((args.source / "config.json").read_text())
    metadata = json.loads((args.sidecar / "qwen36_modelslim_w4a8_metadata.json").read_text())
    text = config.get("text_config", config)
    expected = int(text["num_hidden_layers"]) * int(text["num_experts"]) * 3
    if metadata.get("expert_projection_count") != expected:
        raise ValueError("sidecar does not cover all model expert projections")
    args.output.mkdir(parents=True, exist_ok=True)
    config_target = args.output / "config.json"
    if config_target.is_symlink():
        config_target.unlink()
    link_tree(args.source, args.output)
    for item in args.sidecar.glob("*.w4a8.safetensors"):
        target = args.output / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(item, target)
    source_index = json.loads((args.source / "model.safetensors.index.json").read_text())
    weight_map = {
        key: value for key, value in source_index["weight_map"].items()
        if ".mlp.experts.gate_up_proj" not in key and ".mlp.experts.down_proj" not in key
    }
    expert_keys = set()
    for item in sorted(args.sidecar.glob("*.w4a8.safetensors")):
        with safe_open(str(item), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(f"duplicate merged tensor key: {key}")
                weight_map[key] = item.name
                expert_keys.add(key)
    required = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.{kind}"
        for layer in range(int(text["num_hidden_layers"]))
        for expert in range(int(text["num_experts"]))
        for projection in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale", "weight_offset")
    }
    if expert_keys != required:
        raise ValueError(f"sidecar tensor coverage mismatch: missing={len(required - expert_keys)}, extra={len(expert_keys - required)}")
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "qwen36_modelslim_ssz_w4a8_hybrid_v1"}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n"
    )
    config.setdefault("text_config", {})["ascend_expert_quantization"] = {
        key: value for key, value in metadata.items() if key not in {"files", "expert_projection_count"}
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sidecars": len(list(args.sidecar.glob("*.w4a8.safetensors"))), "merged_tensors": len(weight_map), "format": metadata["format"]}, indent=2))


if __name__ == "__main__":
    main()
