"""独立 ACL 初始化和内存探测，不加载模型或 PyTorch。"""
import ctypes
import faulthandler

faulthandler.dump_traceback_later(15)
print("loading_acl", flush=True)

acl = ctypes.CDLL("libascendcl.so")
print("loaded_acl", flush=True)
print("aclInit", acl.aclInit(None), flush=True)
count = ctypes.c_uint32()
print("device_count", acl.aclrtGetDeviceCount(ctypes.byref(count)), count.value, flush=True)
result = acl.aclrtSetDevice(0)
print("set_device", result, flush=True)
if result == 0:
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    print("memory", acl.aclrtGetMemInfo(0, ctypes.byref(free), ctypes.byref(total)), free.value, total.value, flush=True)
    ptr = ctypes.c_void_p()
    result = acl.aclrtMalloc(ctypes.byref(ptr), 1024 * 1024, 0)
    print("malloc_1MiB", result, flush=True)
    if result == 0:
        print("free", acl.aclrtFree(ptr), flush=True)
    print("reset", acl.aclrtResetDevice(0), flush=True)
print("finalize", acl.aclFinalize(), flush=True)
