import io
import json
import unittest
from decode_trace_utils import trace_events,relative_timestamp,union_duration,intersection_duration,subtract

class TraceTests(unittest.TestCase):
    def test_stream_array_and_object_chunk_boundaries(self):
        rows=[{'name':'模块','args':{'shape':[1,2048]},'ts':'1793780000000000.125'},{'ph':'X','dur':1.25}]
        for payload in [rows,{'schemaVersion':1,'traceEvents':rows,'tail':'ignored'}]:
            self.assertEqual(list(trace_events(io.StringIO(json.dumps(payload,ensure_ascii=False)),7)),rows)
    def test_timestamp_precision(self):
        self.assertEqual(relative_timestamp('1793780000000000.125',1793780000000000),.125)
    def test_overlap_not_double_counted(self):
        intervals=[(0,4),(2,6),(8,10)]
        self.assertEqual(union_duration(intervals),8)
        self.assertEqual(intersection_duration(intervals,[(3,9)]),4)
        self.assertEqual(subtract([(0,10)],intervals),[(6,8)])


class EagerMatmulCategoryTests(unittest.TestCase):
    def test_generic_eager_name_is_disambiguated_by_real_nz_weight_shape(self):
        from decode_trace_utils import kernel_category
        def row(weight):return {'Op Name':'MatMul4/v2','Input Shapes':'"128,1,16,16;'+weight+'"'}
        self.assertEqual(kernel_category(row('128,384,16,16')),'fp16_nz_projection')
        self.assertEqual(kernel_category(row('128,7760,16,16')),'lm_head_fp16')
        self.assertEqual(kernel_category({'Name':'MatMul4/v2','Input Shapes':'N/A'}),'lm_head_fp16')
        with self.assertRaises(ValueError):kernel_category(row('bad'))

class InterruptedArrayTests(unittest.TestCase):
    def test_opt_in_recovers_only_missing_array_close(self):
        import io
        from decode_trace_utils import trace_events
        self.assertEqual(list(trace_events(io.StringIO('[{"ph":"X"}'),allow_missing_final_bracket=True)),[{'ph':'X'}])
        with self.assertRaises(ValueError):list(trace_events(io.StringIO('[{"ph":"X"}')))
        with self.assertRaises(ValueError):list(trace_events(io.StringIO('[{"ph":'),allow_missing_final_bracket=True))
        with self.assertRaises(ValueError):list(trace_events(io.StringIO('[{"ph":"X"},'),allow_missing_final_bracket=True))
        with self.assertRaises(ValueError):list(trace_events(io.StringIO('['),allow_missing_final_bracket=True))

if __name__=='__main__':unittest.main()
