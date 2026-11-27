"""Dispatch independent WebSocket sessions to the 310P AsyncLLM scheduler."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from itertools import count


def build_async_asr(model_class, model_path):
    from qwen_asr.inference.qwen3_asr import Qwen3ASRProcessor
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=model_path, dtype="float16", enforce_eager=True,
        gpu_memory_utilization=0.4, max_model_len=4096,
        max_num_seqs=4, max_num_batched_tokens=2048,
        enable_prefix_caching=False, mm_processor_cache_gb=0,
        async_scheduling=True, disable_log_stats=False,
        worker_extension_cls="int8_worker.Int8WorkerExtension",
    ))
    try:
        # Match the SDK constructor order so the engine's config registration wins.
        processor = Qwen3ASRProcessor.from_pretrained(model_path, fix_mistral_regex=True)
        return model_class(backend="vllm", model=engine, processor=processor,
                           sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                           max_new_tokens=None)
    except BaseException:
        engine.shutdown()
        raise


def advance(continuation, outputs):
    try:
        return False, continuation.send(outputs)
    except StopIteration as done:
        return True, done.value


class AsyncLLMDispatch:
    def __init__(self):
        self.slots = asyncio.Semaphore(8)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r2t2-asr-prep")
        self.ids = count()
        self.calls = set()
        self.closed = False
        self.engine = None

    async def generate(self, prompt, sampling):
        engine_id = f"r2t2-ws-{next(self.ids)}"
        final = None
        async with aclosing(self.engine.generate(prompt, sampling, engine_id)) as updates:
            async for output in updates:
                if output.finished:
                    final = output
        if final is None:
            raise RuntimeError(f"AsyncLLM returned no final output for {engine_id}")
        return [final]

    async def on_owner(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    async def call(self, method, *args, **kwargs):
        if self.closed:
            raise RuntimeError("async ASR dispatcher is closed")
        task = asyncio.current_task()
        self.calls.add(task)
        try:
            async with self.slots:
                if method.__name__ == "init_streaming_state":
                    pending = asyncio.create_task(self.on_owner(lambda: method(*args, **kwargs)))
                    try:
                        return await asyncio.shield(pending)
                    except asyncio.CancelledError:
                        await asyncio.gather(pending, return_exceptions=True)
                        raise
                continuation = getattr(method.__self__, "_async_" + method.__name__)(*args, **kwargs)
                try:
                    outputs = None
                    while True:
                        done, value = await self.on_owner(advance, continuation, outputs)
                        if done:
                            return value
                        outputs = await self.generate(*value)
                finally:
                    # Close behind any outstanding state mutation on the same owner.
                    await self.on_owner(continuation.close)
        finally:
            self.calls.discard(task)

    async def aclose(self):
        self.closed = True
        pending = tuple(self.calls)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        try:
            if self.engine is not None:
                self.engine.shutdown()
                self.engine = None
        finally:
            self.executor.shutdown()
