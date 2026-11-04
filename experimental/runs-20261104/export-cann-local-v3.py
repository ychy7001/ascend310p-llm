#!/usr/bin/env python3
"""Matched offline CANN exporter, existing SQLite only, one export type at a time."""
import argparse,multiprocessing,os,resource,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--kind',choices=['summary','timeline'],required=True);a=p.parse_args()
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1800,1800))
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.environ['OMP_NUM_THREADS']='1'
analysis=Path('results/decode-attribution-20261104/v3/raw/cpu-profiler-matched/profiler_tool/analysis').resolve();sys.path.insert(0,str(analysis))
from msinterface.msprof_export import ExportCommand
from common_func.profiling_scene import ProfilingScene
from profiling_bean.prof_enum.export_data_type import ExportDataType
from msinterface.msprof_entrance import MsprofEntrance
import msinterface.msprof_export as export_module
from msinterface.msprof_output_summary import MsprofOutputSummary
from common_func.path_manager import PathManager
export_module.run_in_subprocess=lambda function,*args,**kwargs:function(*args,**kwargs)
# Skip the vendor aggregation pools for BOTH kinds; retain native per-device outputs.
MsprofOutputSummary.export=lambda *args:None
PathManager.del_summary_and_timeline_dir=staticmethod(lambda paths:None)

def sequential(self,result_dir):
 mode=ProfilingScene().get_mode()
 for event in self.list_map.get('export_type_list',[]):
  name=event.get('export_type',ExportDataType.INVALID).name
  wanted={'OP_SUMMARY','TASK_TIME'} if a.kind=='summary' else {'TASK_TIME','COMMUNICATION'}
  if name not in wanted:continue
  print('SEQUENTIAL_EXPORT',name,flush=True)
  self._multiprocessing_handle_export_data(event,result_dir,mode)
# All inputs already have complete native SQLite. No recalculation is needed.
assert list(a.capture.glob('device_*/sqlite/ai_core_op_summary.db')), 'Missing analyzed input database'
ExportCommand._start_calculate=lambda *args:None
def forbid_child_process(*args,**kwargs):
 raise RuntimeError('CPU offline export must remain serial; vendor child process refused')
multiprocessing.process.BaseProcess.start=forbid_child_process
multiprocessing.Pool=forbid_child_process
ExportCommand._handle_export=sequential
ExportCommand._check_export_timeline_with_so=lambda *args:False
ExportCommand._check_export_summary_with_so=lambda *args:False
sys.argv=['msprof.py','export',a.kind,'-dir',str(a.capture.resolve())]
MsprofEntrance().main()
