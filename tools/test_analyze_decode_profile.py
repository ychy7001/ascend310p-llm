import unittest
from analyze_decode_profile import stable_window
class ProfileWindowTests(unittest.TestCase):
    def test_excludes_mixed_prefill_and_startup_bubbles(self):
        times=[0,100000,200000]+[230000+35000*i for i in range(115)]
        rows=[{'start':t,'category':'lm_head_fp16'} for t in times]
        lo,hi,n,selection=stable_window(rows,96,True)
        self.assertEqual(n,96)
        self.assertAlmostEqual((hi-lo)/n,35000)
        self.assertGreater(lo,200000)
    def test_rejects_tiny_capture(self):
        with self.assertRaises(ValueError):stable_window([{'start':0,'category':'lm_head_fp16'}])

class HardwareMetadataTests(unittest.TestCase):
    def test_trace_recovers_real_graph_id_when_csv_uses_invalid_model(self):
        import csv,json,tempfile
        from pathlib import Path
        from analyze_decode_profile import analyze
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'kernels.csv';trace=root/'trace.json'
            events=[];epoch=1793678979000000
            with source.open('w') as f:
                w=csv.DictWriter(f,fieldnames=['Name','Start Time(us)','Duration(us)','Model ID','Stream ID','Task ID'])
                w.writeheader()
                for i in range(110):
                    start=epoch+i*35000
                    w.writerow({'Name':'MatMul4/v2','Start Time(us)':str(start),'Duration(us)':'10'})
                    w.writerow({'Name':'MatMulV2_NZ_NZ_FP16_test','Start Time(us)':str(start+100),'Duration(us)':'20','Model ID':'4294967295','Stream ID':'17','Task ID':'5'})
                    events.append({'ph':'X','name':'MatMulV2_NZ_NZ_FP16_test','ts':str(start+100),'dur':20,'args':{'Task Type':'AI_CORE','Model Id':1,'Task Id':5,'Physic Stream Id':17}})
                    for j in range(2):events.append({'ph':'X','name':'REDUCE_ASYNC_V2','ts':str(start+200+j*10),'dur':1,'args':{'Task Type':'REDUCE_ASYNC_V2','Model Id':1,'Task Id':30+j,'Physic Stream Id':23}})
            trace.write_text(json.dumps(events));result,rows=analyze(source,trace)
            self.assertEqual(result['graph_fp16_node_count'],1)
            self.assertEqual(result['fp16_hardware_metadata_matches'],96)
            self.assertEqual(result['graph_reduction_counts_by_step'],{2:96})
            self.assertEqual(len(result['graph_reduction_node_ledger']),2)
            self.assertTrue(all(r['count']==96 for r in result['graph_reduction_node_ledger']))

if __name__=='__main__':unittest.main()
