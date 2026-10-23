import logging
from vllm_ascend.worker import utils, model_runner_v1
from vllm_ascend._310p import model_runner_310p
from vllm_ascend.worker.block_table import BlockTable
from vllm.v1.utils import CpuGpuBuffer

def install():
    def snapshot(buffer):
        return buffer.gpu.copy_(buffer.cpu.clone().pin_memory(), non_blocking=False)
    utils.copy_snapshot_to_gpu = snapshot
    model_runner_v1.copy_snapshot_to_gpu = snapshot
    model_runner_310p.copy_snapshot_to_gpu = snapshot
    def blocks(self, num_reqs):
        self.block_table.gpu[:num_reqs].copy_(self.block_table.cpu[:num_reqs], non_blocking=False)
    BlockTable.commit_block_table = blocks
    def copy(self, n=None):
        cpu, gpu = self.cpu, self.gpu
        if n is not None:
            cpu, gpu = cpu[:n], gpu[:n]
        return gpu.copy_(cpu, non_blocking=False)
    CpuGpuBuffer.copy_to_gpu = copy
    logging.warning('clt deployment: synchronous metadata uploads enabled')
