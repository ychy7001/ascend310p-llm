"""Mounted as sitecustomize; CPU scheduler observation only, no model/kernel patch."""
import os
if os.environ.get('GPU_SCHEDULE_PROBE'):
    import json,time
    from vllm.v1.core.sched.scheduler import Scheduler
    _schedule = Scheduler.schedule
    def observed(self, *args, **kwargs):
        out = _schedule(self, *args, **kwargs)
        if out.num_scheduled_tokens:
            row={'time_ns':time.time_ns(),'pid':os.getpid(),'tokens':out.num_scheduled_tokens,
                 'requests':{k:{'output_tokens':self.requests[k].num_output_tokens,
                                'prompt_tokens':self.requests[k].num_prompt_tokens}
                             for k in out.num_scheduled_tokens if k in self.requests}}
            with open(os.environ['GPU_SCHEDULE_PROBE'],'a') as f:f.write(json.dumps(row)+'\n')
        return out
    Scheduler.schedule = observed
