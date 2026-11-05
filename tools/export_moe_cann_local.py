#!/usr/bin/env python3
"""Parse and export raw MoE CANN captures locally, one export type at a time.

Use the analysis package from the exact CANN version that collected the data.
Invoke under the shared CPU-heavy-task flock. Native parsing/calculation remain
intact; vendor export aggregation and process fanout are disabled. The outputs
are the native per-device summary/timeline files, rather than combined views.
"""
import argparse
import multiprocessing
import os
from pathlib import Path
import resource
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--kind', choices=['summary', 'timeline'], required=True)
    args = parser.parse_args()
    analysis = args.analysis_root.resolve(strict=True)
    capture = args.capture.resolve(strict=True)
    if not (analysis / 'msinterface/msprof_export.py').is_file():
        raise ValueError('Expected the matching profiler_tool/analysis directory')
    if not list(capture.glob('device_*/data')):
        raise ValueError('Expected a raw PROF capture with device data directories')
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    os.environ['OMP_NUM_THREADS'] = '1'
    os.umask(0o027)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(analysis))
    from msinterface.msprof_export import ExportCommand
    from msinterface.msprof_entrance import MsprofEntrance
    from msinterface.msprof_output_summary import MsprofOutputSummary
    from common_func.path_manager import PathManager
    from common_func.profiling_scene import ProfilingScene
    from common_func.ms_multi_process import MsMultiProcess
    from profiling_bean.prof_enum.export_data_type import ExportDataType
    import msinterface.msprof_export as export_module
    import msinterface.msprof_c_interface as c_interface
    import common_func.ms_multi_process as multi_process

    def inline(function, *a, **kw):
        return function(*a, **kw)

    export_module.run_in_subprocess = inline
    c_interface.run_in_subprocess = inline
    multi_process.run_in_subprocess = inline
    MsprofOutputSummary.export = lambda *a: None
    PathManager.del_summary_and_timeline_dir = staticmethod(lambda paths: None)

    def sequential(self, result_dir):
        mode = ProfilingScene().get_mode()
        wanted = ({'OP_SUMMARY', 'TASK_TIME'} if args.kind == 'summary'
                  else {'TASK_TIME', 'COMMUNICATION'})
        for event in self.list_map.get('export_type_list', []):
            name = event.get('export_type', ExportDataType.INVALID).name
            if name in wanted:
                print('SEQUENTIAL_EXPORT', name, flush=True)
                self._multiprocessing_handle_export_data(event, result_dir, mode)

    def serial_start(self):
        self.process_init()
        self.ms_run()

    MsMultiProcess.start = serial_start
    MsMultiProcess.join = lambda *a, **kw: None

    def forbid_child_process(*a, **kw):
        raise RuntimeError('Local CANN export must remain serial')

    multiprocessing.process.BaseProcess.start = forbid_child_process
    multiprocessing.Pool = forbid_child_process
    ExportCommand._handle_export = sequential
    ExportCommand._check_export_timeline_with_so = lambda *a: False
    ExportCommand._check_export_summary_with_so = lambda *a: False
    sys.argv = ['msprof.py', 'export', args.kind, '-dir', str(capture)]
    MsprofEntrance().main()
    if not list(capture.glob('device_*/' + args.kind + '/*')):
        raise RuntimeError('CANN did not produce native per-device ' + args.kind)


if __name__ == '__main__':
    main()
