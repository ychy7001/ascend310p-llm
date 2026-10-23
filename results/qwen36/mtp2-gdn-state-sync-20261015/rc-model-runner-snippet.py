        # Temporary rewind guard for KV-load-failure recompute.
        # This can be removed after the upstream fix is merged.
        req_data = scheduler_output.scheduled_cached_reqs

        if self.use_async_scheduling:
            for i, req_id in enumerate(req_data.req_ids):
                req_state = self.requests.get(req_id)
                if req_state is None:
                    continue

                num_computed_tokens = req_data.num_computed_tokens[i]
                if num_computed_tokens < req_state.num_computed_tokens:
                    req_state.prev_num_draft_len = 0

        return super()._update_states(scheduler_output)

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update cached hybrid-model states after model execution."""
        if not self.use_async_scheduling:
            return super()._update_states_after_model_execute(
                output_token_ids, scheduler_output
            )

        if not self.speculative_config or not self.model_config.is_hybrid:
            return

        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)

        if self.cache_config.mamba_cache_mode == "align":
            mamba_utils.postprocess_mamba_align_gpu(
                bufs=self._get_mamba_bufs(),
                num_reqs=num_reqs,
                num_accepted_tokens_gpu=self.num_accepted_tokens.gpu,
                num_accepted_tokens_cpu_tensor=self.num_accepted_tokens.cpu,
                input_batch=self.input_batch,
                kv_cache_config=self.kv_cache_config,
                forward_context=self.compilation_config.static_forward_context,
                mamba_state_copy_funcs=self.model.get_mamba_state_copy_func(),
            )
        else:
            self.num_accepted_tokens.copy_to_cpu(num_reqs)

            if self.cache_config.mamba_cache_mode == "all":
                mamba_utils.postprocess_mamba_all(
                    scheduler_output,
                    self.kv_cache_config,
                    self.input_batch,
                    self.requests,
                    self.mamba_state_idx,
                    self.num_spec_tokens,
                    num_reqs,
                )

        assert self.num_accepted_tokens_event is not None
        self.num_accepted_tokens_event.record()

    def _remap_num_accepted_tokens(self, num_reqs: int) -> None:
        """Map the previous iteration's accepted counts to current rows."""
        prev_idx = self.prev_positions.np[:num_reqs]
        new_mask = prev_idx < 0
        accepted = self.num_accepted_tokens.np
        accepted[:num_reqs] = accepted[np.where(new_mask, 0, prev_idx)]
        accepted[:num_reqs][new_mask] = 1
        self.input_batch.num_accepted_tokens_cpu[:num_reqs] = accepted[:num_reqs]

    def _sync_num_accepted_tokens(
        self, num_reqs: int, has_prev_mapping: bool
    ) -> None:
        """Sync accepted counts while preserving their scheduling-mode owner."""
        if self.use_async_scheduling:
            if has_prev_mapping:
                self._remap_num_accepted_tokens(num_reqs)
            else:
                self.num_accepted_tokens.np[:num_reqs].fill(1)
                self.input_batch.num_accepted_tokens_cpu[:num_reqs].fill(1)
        else:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )

    def _pad_query_start_loc_for_fia(
        self,
        query_start_loc: torch.Tensor,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> int:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
