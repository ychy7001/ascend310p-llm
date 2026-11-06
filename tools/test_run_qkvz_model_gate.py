import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_qkvz_model_gate import Gate, SERVICE, sha

REPO = Path(__file__).resolve().parents[1]


class GateControllerTest(unittest.TestCase):
    def make_gate(self, root):
        gate = Gate.__new__(Gate)
        gate.root = root
        gate.tools = root/'tools'
        gate.tools.mkdir()
        for name in ('eval_gsm8k_http.py', 'check_cache_request_reuse.py'):
            shutil.copyfile(REPO/'tools'/name, gate.tools/name)
        shutil.copyfile(REPO/'benchmarks/ascend310p/check_action_state_regression.py',
                        gate.tools/'check_action_state_regression.py')
        gate.args = SimpleNamespace(arm='A', dataset=root/'missing-dataset', fields=root/'missing-fixtures')
        gate.container = None
        gate.container_arm = None
        gate.started = time.monotonic()
        gate.started_unix_ns = time.time_ns()
        gate.error = None
        return gate

    def snapshot(self, primary='inactive', containers=''):
        return {'services': {SERVICE: {'stdout': primary},
                             'qwen36-image.service': {'stdout': 'inactive'}},
                'containers': {'returncode': 0, 'stdout': containers},
                'user_containers': {'returncode': 0, 'stdout': ''},
                'npu': {'returncode': 0, 'stdout': 'No running processes found'}}

    def test_actual_evaluator_parsers_without_http_or_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = self.make_gate(Path(directory))
            # The inputs intentionally do not exist; successful parsing must not
            # load them or contact the endpoint.
            gate.preflight_evaluators()
            self.assertEqual(len(list(gate.root.glob('A-cli-preflight-*.log'))), 3)
            # Reproduce the older deployed parser that rejected --model.
            path = gate.tools/'check_action_state_regression.py'
            text = path.read_text()
            text = '\n'.join(line for line in text.splitlines()
                             if "p.add_argument('--model'" not in line)+'\n'
            path.write_text(text)
            for file in gate.root.glob('A-cli-preflight-*'):
                file.unlink()
            gate.preflight_evaluators()

    def test_preflight_rejects_incompatible_deployed_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = self.make_gate(Path(directory))
            path = gate.tools/'eval_gsm8k_http.py'
            path.write_text(path.read_text().replace("p.add_argument('--model',required=True)", ''))
            with self.assertRaises(subprocess.CalledProcessError):
                gate.preflight_evaluators()

    def test_failure_cleans_window_without_starting_service(self):
        for initial in ('active', 'inactive'):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as directory:
                gate = self.make_gate(Path(directory))
                with patch.object(gate, 'preflight_evaluators'), \
                     patch.object(gate, 'snapshot', side_effect=[self.snapshot(initial), self.snapshot(), self.snapshot()]), \
                     patch.object(gate, 'launch', side_effect=RuntimeError('startup failed')), \
                     patch.object(gate, 'stop') as stop, \
                     patch('run_qkvz_model_gate.subprocess.run') as run:
                    with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                        gate.execute()
                stop.assert_called_once()
                calls = [call.args[0] for call in run.call_args_list]
                self.assertFalse(any('start' in command for command in calls))
                self.assertEqual(any('stop' in command for command in calls), initial == 'active')
                self.assertTrue(json.loads((gate.root/'A-window-closed.json').read_text())['passed'])

    def test_residual_container_blocks_successful_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = self.make_gate(Path(directory))
            with patch.object(gate, 'preflight_evaluators'), \
                 patch.object(gate, 'snapshot', side_effect=[self.snapshot(), self.snapshot(), self.snapshot(containers='residual')]), \
                 patch.object(gate, 'launch', side_effect=RuntimeError('startup failed')), \
                 patch.object(gate, 'stop'):
                with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                    gate.execute()
            self.assertFalse(json.loads((gate.root/'A-window-closed.json').read_text())['passed'])

    def test_b_preserves_censored_noise_verdict_and_rejects_changed_freeze(self):
        for tampered in (False, True):
            with self.subTest(tampered=tampered), tempfile.TemporaryDirectory() as directory:
                gate = self.make_gate(Path(directory))
                gate.args.arm = 'B'
                gate.rule = {'multiplier': 1, 'absolute_floor': 0}
                noise = {'mode': 'generated', 'metrics': {'max_absolute_target_logprob_difference': 1},
                         'target_logprob_censoring': {'count': 1}}
                paired = {'mode': 'generated', 'metrics': {'max_absolute_target_logprob_difference': 2},
                          'target_logprob_censoring': {'count': 0}, 'scope': 'common prefix only'}
                files = {'A-G1.json': {}, 'A_prime-G1.json': {}, 'inputs.json': {},
                         'threshold-rule.json': gate.rule, 'A_prime-vs-A.json': noise,
                         'A-G2-counts.json': {'GSM64': 63}, 'A_prime-G2-counts.json': {'GSM64': 63}}
                for name, value in files.items():
                    (gate.root/name).write_text(json.dumps(value))
                frozen = {'thresholds': noise['metrics']}
                for key, name in [('A', 'A-G1.json'), ('A_prime', 'A_prime-G1.json'),
                                  ('input', 'inputs.json'), ('rule', 'threshold-rule.json')]:
                    frozen[key+'_sha256'] = sha(gate.root/name)
                frozen['controller_sha256'] = sha(REPO/'tools/run_qkvz_model_gate.py')
                frozen['comparator_sha256'] = sha(REPO/'tools/compare_token_logprobs.py')
                (gate.root/'concrete-thresholds.json').write_text(json.dumps(frozen))
                if tampered:
                    (gate.root/'A-G1.json').write_text('{"changed": true}')
                with patch.object(gate, 'preflight_evaluators'), \
                     patch.object(gate, 'snapshot', side_effect=[self.snapshot()]*3), \
                     patch.object(gate, 'launch') as launch, \
                     patch.object(gate, 'g1', return_value={}), \
                     patch.object(gate, 'g2', return_value={'GSM64': 63}), \
                     patch.object(gate, 'stop'), \
                     patch('run_qkvz_model_gate.compare', return_value=paired):
                    if tampered:
                        with self.assertRaisesRegex(RuntimeError, 'AssertionError'):
                            gate.execute()
                        launch.assert_not_called()
                    else:
                        gate.execute()
                        verdict = json.loads((gate.root/'G1-adjudication.json').read_text())
                        self.assertIsNone(verdict['passed'])
                        self.assertIsNone(verdict['checks']['max_absolute_target_logprob_difference']['passed'])
                        launch.assert_called_once_with('B', candidate=True)


if __name__ == '__main__':
    unittest.main()
