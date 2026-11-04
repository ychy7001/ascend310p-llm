import unittest
from attribute_decode_kernels import attribute

class OrderedAttributionTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for start in (0,100,200):
            rows.append({'start':start,'category':'lm_head_fp16'})
            for i in range(2):
                rows.append({'start':start+10+i,'category':'fp16_nz_projection','duration':i+1,
                             'name':'kernel'+str(i),'model':'N/A','stream':'1','task':str(i),'shapes':'N/A'})
        return sorted(rows,key=lambda r:r['start'])
    def test_join_preserves_group_budget(self):
        result=attribute(self.rows(),[{'group':'a'},{'group':'b'}],0,200,2)
        self.assertAlmostEqual(result['total_ms_per_step'],.003)
        self.assertEqual(result['matched_fp16_calls_per_step'],2)
    def test_missing_or_reordered_kernel_rejected(self):
        rows=self.rows();rows[5]['name']='unexpected'
        with self.assertRaises(ValueError):attribute(rows,[{'group':'a'},{'group':'b'}],0,200,2)
        with self.assertRaises(ValueError):attribute(self.rows(),[{'group':'a'}],0,200,2)


class NativeWeightShapeTests(unittest.TestCase):
    def test_checks_actual_nz_shape_against_runtime_weight(self):
        from attribute_decode_kernels import validate_weight_shape
        self.assertTrue(validate_weight_shape('"128,1,16,16;128,384,16,16"',[6144,2048]))
        self.assertFalse(validate_weight_shape('N/A',[6144,2048]))
        with self.assertRaises(ValueError):validate_weight_shape('"128,1,16,16;128,128,16,16"',[6144,2048])

if __name__=='__main__':unittest.main()
