import ast
import pathlib
import sys
import types
import unittest
from typing import Any
import torch
import torch.nn.functional as F

def module(name, **attrs):
    result = sys.modules[name] = types.ModuleType(name)
    vars(result).update(attrs)
    return result

src = pathlib.Path(__file__).parent
module('torch_npu', npu_format_cast=lambda t, _: t)
module('tests.ut.base', TestBase=unittest.TestCase)
module('vllm_ascend.attention.attention_v1', AscendMetadata=Any)
utils = module('vllm_ascend.utils', torch=torch, F=F, ACL_FORMAT_FRACTAL_NZ=29)
tree = ast.parse((src/'vllm_ascend/utils.py').read_text())
functions = {'nd_to_nz_spec', 'nd_to_nz_2d', '_round_up', '_custom_pad', '_custom_reshape', '_custom_transpose'}
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, 'name', None) in functions], type_ignores=[]), str(src/'vllm_ascend/utils.py'), 'exec'), vars(utils))
for name, p in [('vllm_ascend._310p.attention.attention_mask','vllm_ascend/_310p/attention/attention_mask.py'), ('candidate_tests','tests/ut/_310p/attention/test_attention_mask_310.py')]:
    mod = module(name)
    exec(compile((src/p).read_text(), str(src/p), 'exec'), vars(mod))
runner = unittest.TextTestRunner(verbosity=2)
raise SystemExit(not runner.run(unittest.defaultTestLoader.loadTestsFromName('candidate_tests')).wasSuccessful())
