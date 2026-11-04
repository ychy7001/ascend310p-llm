# SPDX-License-Identifier: Apache-2.0
"""9.2.1固定默认模块，仅替换PR17640配套H/O；保留Torch WY。"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import torch
from vllm_ascend.utils import enable_custom_op

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--probe', choices=['repeat', 'strict'], required=True)
a, arguments = p.parse_known_args()
assert enable_custom_op()
m = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
actual = hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
if actual != '4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f':
    raise RuntimeError('Unexpected default9.2.1 GDN source: ' + actual)
assert not hasattr(m, '_production_wy_before_native')
wy = m._compute_kernel_inputs_from_torch_wy
# NPUBridge snapshots this search order during dlopen. vLLM bootstrap above
# prepends its default vendor, which exports the same API names on9.2.1.
vendor = a.library.resolve().parent.parent / 'opp/vendors/hyperqwen_gdn_pair921_transformer'
api = vendor / 'op_api/lib/libcust_opapi.so'
if not api.is_file():
    raise RuntimeError('Missing isolated paired API: ' + str(api))
paths = [str(vendor)] + [x for x in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':')
                         if x and x != str(vendor)]
os.environ['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(paths)
print(json.dumps(dict(event='paired_api_search_order_before_dlopen', paths=paths,
                      paired_api_sha256=hashlib.sha256(api.read_bytes()).hexdigest())), flush=True)
torch.ops.load_library(str(a.library))
torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h_vllm = torch.ops.pr17640_gdn_pair.fwd_h
torch.ops._C_ascend.chunk_fwd_o_vllm = torch.ops.pr17640_gdn_pair.fwd_o
print(json.dumps(dict(scope='Pinned PR17640 H/O pair on9.2.1; original Torch WY unchanged',
                      library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
                      chunk_sha256=actual)), flush=True)
name = 'repro_chunk_gdn_nondeterminism.py' if a.probe == 'repeat' else 'check_default_gdn.py'
script = Path(__file__).parent / name
sys.argv = [str(script), *arguments]
runpy.run_path(str(script), run_name='__main__')
assert m._compute_kernel_inputs_from_torch_wy is wy
assert not hasattr(m, '_production_wy_before_native')

loaded_api_paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                           if 'libcust_opapi.so' in line})
if str(api.resolve()) not in loaded_api_paths:
    raise RuntimeError('Isolated paired API not mapped after execution: ' + repr(loaded_api_paths))
print(json.dumps(dict(event='paired_api_loaded', loaded_api_paths=loaded_api_paths)), flush=True)
