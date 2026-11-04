# SPDX-License-Identifier: Apache-2.0
"""在模型加载前安装 WY 实验入口；使用生产 310P worker。"""
from vllm_ascend._310p.worker_310p import NPUWorker310


class NativeWYWorker(NPUWorker310):
    def init_device(self):
        super().init_device()
        from native_wy import install
        install()
