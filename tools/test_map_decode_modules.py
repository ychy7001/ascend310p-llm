import unittest
from map_decode_modules import module_path,reaches_output,module_group
class ModuleMapTests(unittest.TestCase):
    def test_inlined_module_stack(self):
        row={'module_stack':{'last':'("L[\'self\'].layers.0.linear_attn.out_proj", <class \'Linear\'>)'}}
        name=module_path(row)
        self.assertEqual(name,'language_model.model.layers.0.linear_attn.out_proj')
        self.assertEqual(module_group(name),'linear_out_proj')
    def test_dependency_chain_and_unused_reduction(self):
        rows=[{'name':'reduce','op':'call_function','args':'(input,)','kwargs':'{}'},
              {'name':'dead','op':'call_function','args':'(input,)','kwargs':'{}'},
              {'name':'project','op':'call_function','args':'(reduce,)','kwargs':'{}'},
              {'name':'output','op':'output','args':'(project,)','kwargs':'{}'}]
        reach=reaches_output(rows)
        self.assertTrue(reach['reduce']);self.assertFalse(reach['dead'])
if __name__=='__main__':unittest.main()
