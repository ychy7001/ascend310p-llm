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

if __name__=='__main__':unittest.main()
