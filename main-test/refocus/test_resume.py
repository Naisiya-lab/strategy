"""CPU tests: recovery, immutable completed items, retries, and input identity."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import evaluate as ev


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ev.ROOT)
        self.root = Path(self.tmp.name)
        images = self.root / 'data' / 'images'
        images.mkdir(parents=True)
        rows = []
        for i in range(3):
            (images / f'{i}.png').write_bytes(b'CPU fixture, not a real image')
            rows.append(dict(item_id=str(i), image_id=str(i), input=f'Question {i}',
                             task_type='TQA', dataset_name='WTQ'))
        ev.save(self.root / 'data' / 'ann.json', rows)
        self.cfg = dict(data_root=str(images.parent), annotation='ann.json', seed=42,
                        gpu_ids=['0', '1'], max_tokens=1024)
        ev.save(self.root / 'config.json', self.cfg)
        self.out = self.root / 'outputs' / 'mock-smoke'

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args, success=True):
        r = subprocess.run([sys.executable, str(ev.ROOT / 'evaluate.py'), 'smoke', '--mock',
                            '--config', str(self.root / 'config.json'),
                            '--output-root', str(self.root / 'outputs'), *args],
                           capture_output=True, text=True, timeout=60)
        if success:
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        else:
            self.assertNotEqual(r.returncode, 0)
        return r

    def records(self):
        return sorted((self.out / 'samples').glob('*.json'))

    def test_resume_partial_and_no_duplicate_export(self):
        self.run_cli()
        files = self.records()
        kept = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files[:2]}
        files[2].unlink()  # Simulate an unfinished item after interruption.
        (files[2].with_suffix('.tmp')).write_text('incomplete write')
        self.run_cli('--gpus', '2')  # Hardware allocation may change on resume.
        for p, previous in kept.items():
            self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), previous)
        self.assertEqual(json.loads((self.out / 'summary.json').read_text())['newly_attempted'], 1)
        self.run_cli()
        self.assertEqual(json.loads((self.out / 'summary.json').read_text())['newly_attempted'], 0)
        lines = (self.out / 'predictions.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(len({json.loads(x)['question_id'] for x in lines}), 3)

    def test_error_retry_requires_flag_and_archives(self):
        self.run_cli()
        p = self.records()[0]
        record = json.loads(p.read_text())
        record.update(status='error', error='Simulated OOM')
        ev.save(p, record)
        self.run_cli()
        self.assertEqual(json.loads(p.read_text())['status'], 'error')
        self.run_cli('--retry-errors')
        self.assertEqual(json.loads(p.read_text())['status'], 'ok')
        self.assertEqual(len(list((self.out / 'previous_attempts').glob('*.json'))), 1)

    def test_changed_config_refused(self):
        self.run_cli()
        self.cfg['max_tokens'] = 2048
        ev.save(self.root / 'config.json', self.cfg)
        r = self.run_cli(success=False)
        self.assertIn('Inputs/code/environment changed', r.stderr)

    def test_corrupt_result_is_not_silently_overwritten(self):
        self.run_cli()
        p = self.records()[0]
        p.write_text('{bad')
        self.run_cli(success=False)
        self.assertEqual(p.read_text(), '{bad')

    def test_export_partial_without_model_or_config(self):
        self.run_cli()
        self.records()[0].unlink()
        r = subprocess.run([sys.executable, str(ev.ROOT / 'evaluate.py'), 'export',
                            '--run-name', 'mock-smoke', '--output-root', str(self.root / 'outputs')],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads((self.out / 'summary.json').read_text())['missing'], 1)
        self.assertEqual(len((self.out / 'predictions.jsonl').read_text().splitlines()), 2)

    def test_mock_scoring_refused(self):
        self.run_cli()
        r = subprocess.run([sys.executable, str(ev.ROOT / 'score.py'), '--run', str(self.out)],
                           capture_output=True, text=True, timeout=30)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('scoring refused', r.stderr)

    def test_lock_released_after_process_is_killed(self):
        self.out.mkdir(parents=True)
        script = ('import time; from pathlib import Path; from evaluate import run_lock; '
                  'p=Path(__import__("sys").argv[1]); '
                  '\nwith run_lock(p):\n print("LOCKED",flush=True); time.sleep(60)')
        p = subprocess.Popen([sys.executable, '-c', script, str(self.out)],
                             cwd=ev.ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(), 'LOCKED')
            with self.assertRaises(RuntimeError):
                with ev.run_lock(self.out):
                    pass
        finally:
            p.kill(); p.communicate(timeout=10)
        with ev.run_lock(self.out):
            pass


if __name__ == '__main__':
    unittest.main()
