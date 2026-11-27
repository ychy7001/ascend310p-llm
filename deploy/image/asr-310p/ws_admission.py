"""Admission and bounded ingress for async WebSocket sessions."""

import asyncio
from contextlib import asynccontextmanager
from hmac import compare_digest
import json


async def recv(ws, timeout):
    try:
        async with asyncio.timeout(timeout):
            return await ws.recv()
    except asyncio.CancelledError:
        # Sanic uses CancelledError for peer close as well as for task cancellation.
        if asyncio.current_task().cancelling():
            raise
        return None


class StreamGuard:
    capacity = 2
    queued_frames = 16
    max_packet = 32768
    queued_bytes = 81920

    def __init__(self):
        self.active = 0

    async def read_header(self, ws, keys):
        try:
            raw = await recv(ws, 5)
        except TimeoutError:
            await self.reject(ws, None, "header_timeout", 1008)
            return None
        except Exception:
            return None  # Peer closed before sending a header.
        if raw is None:
            return None
        try:
            header = json.loads(raw) if isinstance(raw, str) and len(raw) <= self.max_packet else None
        except ValueError:
            header = None
        if not isinstance(header, dict) or not isinstance(header.get("requestId"), str):
            await self.reject(ws, None, "invalid_header", 1008)
            return None
        key = header.get("secret_key")
        if not isinstance(key, str) or not any(compare_digest(key.encode(), value.encode()) for value in keys):
            await self.reject(ws, header["requestId"], "unauthorized", 4401)
            return None
        return header

    @staticmethod
    async def reject(ws, request_id, reason, code):
        try:
            await ws.send(json.dumps(dict(status="error", requestId=request_id, msg=reason)))
        finally:
            await ws.close(code=code, reason=reason)

    @asynccontextmanager
    async def admit(self, ws, request_id):
        if self.active >= self.capacity:
            await self.reject(ws, request_id, "server_busy", 1013)
            yield False
            return
        self.active += 1
        try:
            yield True
        finally:
            self.active -= 1

    def queue(self):
        return AudioQueue(self.queued_frames + 1)

    async def receive(self, ws, queue, eos, timeout):
        finished = False
        while True:
            try:
                packet = await recv(ws, timeout)
            except TimeoutError:
                return "receive_timeout"
            except Exception:
                return "disconnected"
            if packet is None:
                return "disconnected"
            if not finished and packet == eos:
                queue.put_nowait(packet)  # The extra slot is reserved for ordered EOS.
                finished = True
                continue  # Keep watching the peer while terminal inference is running.
            if finished or not isinstance(packet, bytes) or not packet:
                return "invalid_audio_frame"
            if len(packet) > self.max_packet:
                return "audio_frame_too_large"
            if queue.qsize() >= self.queued_frames or queue.pcm_bytes + len(packet) > self.queued_bytes:
                return "audio_queue_full"
            queue.put_nowait(packet)

    async def run(self, ws, queue, eos, processor, timeout):
        receiver = asyncio.create_task(self.receive(ws, queue, eos, timeout))
        worker = asyncio.create_task(processor())
        try:
            done, _ = await asyncio.wait((receiver, worker), return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                return await receiver
            await worker
            return "processor_done"
        finally:
            for task in (receiver, worker):
                if not task.done():
                    task.cancel()
            await asyncio.gather(receiver, worker, return_exceptions=True)

    async def finish(self, ws, request_id, reason):
        if reason in {"audio_queue_full", "audio_frame_too_large", "invalid_audio_frame", "receive_timeout"}:
            code = {"audio_queue_full": 1013, "audio_frame_too_large": 1009}.get(reason, 1008)
            await self.reject(ws, request_id, reason, code)
        elif reason != "disconnected":
            await ws.close()


class AudioQueue(asyncio.Queue):
    pcm_bytes = 0

    def put_nowait(self, entry):
        super().put_nowait(entry)
        if isinstance(entry, bytes):
            self.pcm_bytes += len(entry)

    def get_nowait(self):
        entry = super().get_nowait()
        if isinstance(entry, bytes):
            self.pcm_bytes -= len(entry)
        return entry
