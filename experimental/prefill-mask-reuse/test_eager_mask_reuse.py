from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace as Namespace
from unittest import TestCase, main
from unittest.mock import patch
import sys

import eager_mask_reuse as reuse


class Lengths:
    def __init__(self, *values):
        self.values = values

    def to(self, *args, **kwargs):
        return self

    def tolist(self):
        return list(self.values)


class MaskReuseTest(TestCase):
    def test_forward_scope_lifetime(self):
        @reuse.scoped_forward
        def inner():
            return reuse.one_mask('layout', object)

        @reuse.scoped_forward
        def outer():
            mask = reuse.one_mask('layout', object)
            self.assertIsNot(mask, inner())
            self.assertIs(mask, reuse.one_mask('layout', object))
            return mask

        @reuse.scoped_forward
        def failing():
            reuse.one_mask('layout', object)
            raise RuntimeError('interrupted')

        self.assertIsNot(outer(), outer())
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            failing()
        self.assertIsNone(reuse._scope.get())
        self.assertIsNot(reuse.one_mask('layout', object), reuse.one_mask('layout', object))

    def test_installed_patch_matches_ordered_values_and_skips_graph(self):
        state = Namespace(ChunkedPrefill=0, PrefillCacheHit=1, SpecDecoding=2)
        mode = Namespace(NONE=0, FULL=1)
        context = Namespace(capturing=False, cudagraph_runtime_mode=mode.NONE)

        class Builder:
            @classmethod
            def get_splitfuse_mask(cls, metadata, device):
                return object()

        class Runner:
            def _model_forward(self, check):
                return check()

        def mask(queries, contexts, device='npu:0', attn_state=state.ChunkedPrefill):
            return Builder.get_splitfuse_mask(Namespace(
                query_start_loc=Lengths(*queries), seq_lens=Lengths(*contexts),
                attn_state=attn_state,
            ), device)

        modules = {
            'torch': Namespace(device=str, int32='i32', float16='f16'),
            'vllm.config.compilation': Namespace(CUDAGraphMode=mode),
            'vllm.forward_context': Namespace(get_forward_context=lambda: context),
            'vllm_ascend.attention.attention_v1': Namespace(AscendAttentionState=state),
            'vllm_ascend._310p.attention.attention_mask': Namespace(AttentionMaskBuilder310=Builder),
            'vllm_ascend._310p.model_runner_310p': Namespace(NPUModelRunner310=Runner),
        }
        with patch.dict(sys.modules, modules), patch.dict('os.environ', {'QWEN36_EAGER_MASK_REUSE': '1'}), redirect_stdout(StringIO()):
            reuse.install()

        def check():
            def same_layout():
                return mask((0, 64, 192), (4096, 8192))

            first = same_layout()
            self.assertIs(first, mask((0, 64, 192), (4096, 8192), attn_state=state.PrefillCacheHit))
            self.assertIsNot(first, mask((0, 128, 192), (4096, 8192)))
            self.assertIsNot(same_layout(), mask((0, 64, 192), (8192, 4096)))
            next_mask = same_layout()
            self.assertIsNot(first, next_mask)
            self.assertIsNot(next_mask, mask((0, 64, 192), (4096, 8192), device='npu:1'))
            active = same_layout()
            context.capturing = True
            self.assertIsNot(active, same_layout())
            context.capturing = False
            context.cudagraph_runtime_mode = mode.FULL
            self.assertIsNot(active, same_layout())
            context.cudagraph_runtime_mode = mode.NONE
            self.assertIsNot(active, mask((0, 64, 192), (4096, 8192), attn_state=state.SpecDecoding))
            self.assertIs(active, same_layout())
            return active

        runner = Runner()
        self.assertIsNot(runner._model_forward(check), runner._model_forward(check))


if __name__ == '__main__':
    main()
