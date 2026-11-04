import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import observe_decode as obs

class ObservationTests(unittest.TestCase):
    def test_metadata_never_copies_token_values(self):
        req=NS(request_id='private-id', num_computed_tokens=4100,num_prompt_tokens=8192,
               num_output_tokens=0,arrival_time=10,prompt_token_ids=['SECRET'], _output_token_ids=[])
        row=obs.request_meta(req,4096,True)
        self.assertEqual(row['computed'],4)
        self.assertEqual(row['phase'],'prefill')
        self.assertNotIn('SECRET',json.dumps(row))
        self.assertNotIn('private-id',json.dumps(row))

    def test_wrappers_preserve_result_exception_and_scheduler_state(self):
        class Scheduler:
            def __init__(self):
                self.requests={};self.waiting=[];self.running=[];self.current_step=0
            def add_request(self,r): self.requests[r.request_id]=r;return 'added'
            def schedule(self):
                self.current_step+=1
                r=self.requests['id'];r.num_computed_tokens+=2
                return NS(num_scheduled_tokens={'id':2},total_num_scheduled_tokens=2)
            def update_from_output(self,*args): raise ValueError('original-error')
        obs.patch_scheduler(NS(Scheduler=Scheduler))
        scheduler=Scheduler()
        req=NS(request_id='id',num_computed_tokens=0,num_prompt_tokens=4,
               num_output_tokens=0,arrival_time=10,_output_token_ids=[])
        with patch.object(obs,'emit') as emit:
            self.assertEqual(scheduler.add_request(req),'added')
            result=scheduler.schedule()
            self.assertEqual(req.num_computed_tokens,2)
            self.assertEqual(emit.call_args.kwargs['requests'][0]['computed'],0)
            with self.assertRaisesRegex(ValueError,'original-error'):
                scheduler.update_from_output(result,None)

    def test_output_per_process(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'DECODE_OBSERVE_DIR':directory}):
            old=obs._STREAM;obs._STREAM=None
            try:
                obs.emit('test',number=3)
                rows=list(Path(directory).glob('*.jsonl'))
                self.assertEqual(len(rows),1)
                self.assertEqual(json.loads(rows[0].read_text())['number'],3)
            finally:
                obs._STREAM.close();obs._STREAM=old

if __name__=='__main__':unittest.main()
