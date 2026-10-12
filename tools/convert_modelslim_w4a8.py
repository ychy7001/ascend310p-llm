#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stream expert weights from a BF16/FP16 Qwen3.6 checkpoint into W4A8 sidecars.

The source checkpoint is never modified. The output contains only quantized
expert tensors; the original checkpoint remains the source of all non-expert
weights until a loader manifests both sets together.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from modelslim_ssz_adapter import quantize_modelslim_ssz

EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")

def pack_signed(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or q.shape[1] % 2 or q.min() < -8 or q.max() > 7:
        raise ValueError(f"invalid SSZ int4 tensor {tuple(q.shape)} range={q.min().item()}..{q.max().item()}")
    u = q.to(torch.int16)
    return ((u[:, 0::2] & 15) | ((u[:, 1::2] & 15) << 4)).to(torch.int8).contiguous()


def quantize_projection(key: str, weight: torch.Tensor, layer: int, output: dict) -> int:
    """Split Qwen3.6 packed experts and emit the vLLM bank layout."""
    if weight.ndim != 3:
        raise ValueError(f"{key}: expected [experts, output, input], got {tuple(weight.shape)}")
    if weight.shape[0] <= 0:
        raise ValueError(f"{key}: empty expert dimension")
    if key.endswith("gate_up_proj"):
        if weight.shape[1] % 2:
            raise ValueError(f"{key}: gate_up output dimension must be even")
        projections = (("gate_proj", weight[:, : weight.shape[1] // 2]),
                       ("up_proj", weight[:, weight.shape[1] // 2 :]))
    else:
        projections = (("down_proj", weight),)
    count = 0
    for proj, bank in projections:
        for expert, matrix in enumerate(bank):
            matrix = matrix.to(torch.float32)
            q, scale, offset = quantize_modelslim_ssz(matrix, groupwise=False)
            if matrix.shape[1] % 128:
                raise ValueError(f"{key}: K={matrix.shape[1]} is not divisible by G128")
            name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
            output[name + ".weight"] = pack_signed(q)
            output[name + ".weight_scale"] = scale.expand(-1, matrix.shape[1] // 128).contiguous()
            output[name + ".weight_offset"] = offset.expand(-1, matrix.shape[1] // 128).contiguous()
            count += 1
    return count

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    args=ap.parse_args(); src=Path(args.input); dst=Path(args.output); dst.mkdir(parents=True, exist_ok=True)
    files=sorted(src.glob("*.safetensors")); manifest=[]; count=0
    for file in files:
        print(json.dumps({"event": "start_shard", "source": file.name}), flush=True)
        out={}
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                match = EXPERT.fullmatch(key)
                if not match:
                    continue
                count += quantize_projection(key, f.get_tensor(key), int(match.group(1)), out)
        if out:
            target=dst/(file.stem+".w4a8.safetensors")
            save_file(out, str(target), metadata={"format":"qwen36_modelslim_ssz_w4a8_v1","source":file.name})
            manifest.append({"source":file.name,"output":target.name,"tensors":len(out)})
            print(json.dumps({"event": "saved_shard", "source": file.name,
                              "output": target.name, "tensors": len(out),
                              "bytes": target.stat().st_size}), flush=True)
    config = json.loads((src / "config.json").read_text())
    text = config.get("text_config", config)
    expected = int(text["num_hidden_layers"]) * int(text["num_experts"]) * 3
    if count != expected:
        raise ValueError(f"incomplete expert checkpoint: {count} projections, expected {expected}")
    metadata={"format":"qwen36_modelslim_ssz_w4a8_v1","bits":4,"symmetric":True,"packing":"signed_int4_low_nibble_first_in_axis","scale_dtype":"float16","offset_dtype":"float16","backend":"cube_310_int4_a8_modelslim","activation_quantization":"int8_per_token","group_size":128,"scale_layout":"per_output_channel_repeated_g128","source_expert_layout":"qwen3_6_packed_gate_up_down","expert_projection_count":count,"files":manifest}
    (dst/"qwen36_modelslim_w4a8_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (dst/"manifest.json").write_text(json.dumps({"source":str(src),"quantization":metadata},indent=2)+"\n")
    print(json.dumps({"expert_weight_tensors":count,"output":str(dst),"files":manifest},indent=2))
if __name__ == "__main__": main()
