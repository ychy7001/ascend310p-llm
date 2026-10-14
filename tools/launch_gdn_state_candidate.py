"""从已保存的 podman inspect 配置启动独立 GDN 修复候选，不修改原服务。"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--c16", action="store_true", help="隔离单请求 C8 adapter，验证并发 GDN"
    )
    args = parser.parse_args()
    work = args.work.resolve()
    source = (work / "image-wrapper.py").read_text()
    original = "    flat_state_indices = _flatten_state_indices(ssm_state_indices, cu_seqlens, total_tokens)"
    replacement = """    if ssm_state_indices.ndim == 2:
        # Keep the full historical row independently of current query lengths.
        flat_state_indices = ssm_state_indices[:cu_seqlens.shape[0] - 1].to(torch.int32).contiguous()
    else:
        flat_state_indices = _flatten_state_indices(ssm_state_indices, cu_seqlens, total_tokens)"""
    assert source.count(original) == 1
    wrapper = work / "candidate-wrapper.py"
    source = source.replace(original, replacement)
    marker = "    out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310("
    source = source.replace(
        marker,
        """    if _EXTRA_CTX.capturing:
        print("GDN_STATE_TABLE_CAPTURE", tuple(q.shape), tuple(flat_state_indices.shape),
              tuple(actual_seq_lengths.shape), flush=True)
"""
        + marker,
    )
    wrapper.write_text(source)
    info = json.loads((work / "chip1-restore.json").read_text())[0]
    command = info["Config"]["CreateCommand"].copy()
    command[command.index("--name") + 1] = "qwen36-gdn-state-candidate"
    command = ["PORT=18029" if part == "PORT=18028" else part for part in command]
    if args.c16:
        command = [
            "QWEN36_C8_ENABLED=0"
            if part == "QWEN36_C8_ENABLED=1"
            else "MAX_MODEL_LEN=32768"
            if part.startswith("MAX_MODEL_LEN=")
            else part
            for part in command
        ]
    image_pos = command.index(info["Config"]["Image"])
    command[image_pos:image_pos] = [
        "-v",
        f"{work}/opp/vendors/custom_transformer:/recurrent-opp:ro",
        "-v",
        f"{wrapper}:/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/gdn_310.py:ro",
    ]
    script = command[-1]
    marker = '    python3 -c "import sitecustomize; print(sitecustomize.__file__)"'
    assert script.count(marker) == 1
    command[-1] = script.replace(
        marker,
        """    # Register the embedded vendor now so enable_custom_op() does not prepend it.
    export ASCEND_CUSTOM_OPP_PATH="/recurrent-opp:$ASCEND_CUSTOM_OPP_PATH:/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    export LD_LIBRARY_PATH="/recurrent-opp/op_api/lib:$LD_LIBRARY_PATH"
"""
        + marker,
    ).replace("--max-num-seqs 1", "--max-num-seqs 4")
    (work / "candidate-command.json").write_text(json.dumps(command, indent=2))
    with (work / "candidate.log").open("a") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"candidate launcher pid={process.pid}, port=18029")


if __name__ == "__main__":
    main()
