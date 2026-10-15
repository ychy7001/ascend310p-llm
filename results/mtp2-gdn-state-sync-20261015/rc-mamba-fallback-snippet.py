def _postprocess_mamba_align_gpu_cpu_fallback(
    *,
    bufs: "mamba_utils.MambaBuffers",
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
) -> None:
    """CPU fallback for 310P where the Triton fused postprocess is unavailable."""
    ctx = bufs.postprocess_align
    assert ctx is not None
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    # stage_postprocess_inputs_to_gpu has already materialized the same
    # per-request values into the CpuGpuBuffer numpy views. 310P cannot use the
    # Triton fused kernel, so reuse the CPU views to mirror its decision logic.
    mamba_state_idx = ctx.mamba_state_idx_buf.np
    num_scheduled_tokens = ctx.num_scheduled_tokens_buf.np
    num_computed_tokens = ctx.num_computed_tokens_buf.np
    num_draft_tokens = ctx.num_draft_tokens_buf.np
    block_size = ctx.block_size

    # Upstream initializes num_accepted_tokens_out from the real accepted-token
    # counts, then only overwrites entries where src and dest are the same
    # block. Preserve that default so the next preprocess keeps the right
    # accept_token_bias when multiple draft tokens were accepted.
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(num_accepted_tokens_gpu[:num_reqs])
    num_accepted_tokens = input_batch.num_accepted_tokens_cpu
    for i in range(num_reqs):
        num_tokens_running_state = num_computed_tokens[i] + num_scheduled_tokens[i] - num_draft_tokens[i]
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens[i] - 1
        aligned_new_computed_tokens = new_num_computed_tokens // block_size * block_size
        if aligned_new_computed_tokens < num_tokens_running_state:
            continue

        src_block_idx = mamba_state_idx[i]
        dest_block_idx = aligned_new_computed_tokens // block_size - 1
        accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
        if src_block_idx == dest_block_idx:
            # Match the fused kernel: once the running state remains in the
            # same block, the next preprocess should start from token bias 0.
            num_accepted_tokens_cpu_tensor[i] = 1
            if accept_token_bias == 0:
                continue

        # The upstream fused kernel also copies Mamba state in this postprocess
        # step. Do the same with tensor views so 310P avoids Triton without
        # changing where conv/temporal state lands before the next iteration.
        for mamba_group_id in ctx.mamba_group_ids:
            block_ids = input_batch.block_table[mamba_group_id].get_numpy_array()[i]
            dest_block_id = block_ids[dest_block_idx]
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
