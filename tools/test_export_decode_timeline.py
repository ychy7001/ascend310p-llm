import csv,json,tempfile,unittest
from pathlib import Path
from export_decode_timeline import export,kernel_row
class TimelineExportTests(unittest.TestCase):
 def test_preserves_precise_timestamps_and_rejects_wait_as_compute(self):
  event={'ph':'X','name':'MatMul4/v2','ts':'1793784711395392.100','dur':5.547,'args':{'Task Type':'AI_CORE','Model Id':1,'Physic Stream Id':7,'Task Id':3}}
  row=kernel_row(event);self.assertEqual(row['Start Time(us)'],event['ts']);self.assertEqual(row['Model ID'],1)
  self.assertIsNone(kernel_row({**event,'name':'NOTIFY_WAIT','args':{'Task Type':'NOTIFY_WAIT'}}))
  self.assertIsNotNone(kernel_row({**event,'name':'hcom_allReduce__1','args':{'model id':1}}))
  self.assertEqual(row['Input Shapes'],'N/A')
 def test_streams_complete_array_and_fails_truncated_input(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);source=root/'trace.json';target=root/'kernel.csv'
   event={'ph':'X','name':'op','ts':'1.001','dur':2,'args':{'Task Type':'AI_CPU'}}
   source.write_text(json.dumps([event]));report=export([source],target)
   self.assertEqual(report['kernel_rows'],1)
   with target.open() as stream:self.assertEqual(len(list(csv.DictReader(stream))),1)
   source.write_text(json.dumps([event])[:-1])
   with self.assertRaises(ValueError):export([source],target)
if __name__=='__main__':unittest.main()
