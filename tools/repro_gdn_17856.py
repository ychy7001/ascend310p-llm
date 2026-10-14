import json
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C

torch.npu.set_device(0)
op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310
print(op.default._schema)

# Match the upstream tested 128-dimension contract. Constant states and zero
# keys/beta make state writes deterministic: each token halves its input state.
tokens, nk, nv, key_dim, value_dim, slots = 6, 4, 8, 128, 128, 8
torch.manual_seed(17856)
q = torch.full((tokens, nk, key_dim), 1/16, dtype=torch.float16, device="npu")
k = torch.zeros_like(q)
v = torch.zeros((tokens, nv, value_dim), dtype=torch.float16, device="npu")
beta = torch.zeros((tokens, nv), dtype=torch.float16, device="npu")
g = torch.full((tokens, nv), -0.69314718, dtype=torch.float32, device="npu")
cu = torch.tensor([2, 4], dtype=torch.int32, device="npu")
state_indices = torch.tensor([0, 1, 4, 5, 6, 7], dtype=torch.int32, device="npu")

cases = [([2, 1], "control"), ([4, 1], "original_bug"), ([4, 1], "gathered_state")]
records = []
for accepted, mode in cases:
    state = torch.full((slots, nv, value_dim, key_dim), 0.25, dtype=torch.float16, device="npu")
    # Different accepted-history state proves we did not merely clamp the count.
    state[3].fill_(0.5)
    kernel_accepted = accepted
    if mode == "gathered_state":
        # Preserve the accepted state before the current-token state slots are
        # compacted. This emulates an explicit per-request state gather.
        state[0].copy_(state[3])
        kernel_accepted = [1, 1]
    out = op(q, k, v, beta, state, cu, state_indices, g, None,
             torch.tensor(kernel_accepted, dtype=torch.int32, device="npu"), key_dim ** -0.5)
    torch.npu.synchronize()
    record = {
        "accepted": accepted,
        "mode": mode,
        "output_shape": list(out.shape),
        "output_abs_sum": float(out.float().abs().sum().cpu()),
        "state_abs_sum": float(state.float().abs().sum().cpu()),
        "output_per_token": out.cpu().mean(dim=(1,2)).tolist(),
        "state_slot_means": state.cpu().float().mean(dim=(1,2,3)).tolist(),
        "expected_state_slot_means": [0.125, 0.0625, 0.25, 0.5, 0.125, 0.0625, 0.03125, 0.015625] if mode == "control" else [0.25, 0.125, 0.25, 0.5, 0.125, 0.0625, 0.03125, 0.015625],
    }
    print(json.dumps(record), flush=True)
    records.append(record)

assert records[0]["state_slot_means"] == records[0]["expected_state_slot_means"]
assert records[1]["state_slot_means"] != records[1]["expected_state_slot_means"]
assert records[2]["state_slot_means"] == records[2]["expected_state_slot_means"]
