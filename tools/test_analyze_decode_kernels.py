import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class KernelSpanTest(unittest.TestCase):
    def test_both_timestamp_formats_and_overlapping_streams(self):
        script = Path(__file__).with_name('analyze_decode_kernels.py')
        for timestamp in ('Start Time(us)', 'Task Start Time(us)'):
            with self.subTest(timestamp=timestamp), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp) / 'kernels.csv', Path(tmp) / 'summary.json'
                fields = ['Name', 'Duration(us)', timestamp, 'Input Shapes', 'Input Data Types']
                with source.open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for start, duration in [(0, 5), (3, 4), (10, 2)]:
                        writer.writerow({'Name': 'op', 'Duration(us)': duration,
                                         timestamp: start, 'Input Shapes': '1', 'Input Data Types': 'FP16'})
                subprocess.run([sys.executable, str(script), '--input', str(source),
                                '--out', str(output)], check=True, capture_output=True)
                result = json.loads(output.read_text())
                self.assertEqual(result['sum_kernel_us'], 11)
                self.assertEqual(result['device_task_span_us'], 12)
                self.assertEqual(result['captured_kernel_interval_union_us'], 9)
                self.assertEqual(result['captured_kernel_interval_coverage'], .75)


if __name__ == '__main__':
    unittest.main()
