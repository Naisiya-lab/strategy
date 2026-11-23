import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import io
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('evaluate', Path(__file__).with_name('evaluate.py'))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class IntegrationTest(unittest.TestCase):
    def test_http_resume_export_and_manifest_guard(self):
        seen = []
        def mock_open(req, timeout):
            if req.data is None:
                return io.BytesIO(b'{"data":[{"id":"qwen3-vl-32b"}]}')
            body = json.loads(req.data)
            seen.append(body)
            q = body['messages'][0]['content'][1]['text']
            return io.BytesIO(json.dumps({'choices':[{'message':{'content':'answer'},
                'finish_reason':'length' if q == 'long' else 'stop'}],
                'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}}).encode())
        with patch.object(ev.urllib.request, 'urlopen', side_effect=mock_open):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); data = root/'data'; (data/'images'/'nested').mkdir(parents=True)
                (data/'images'/'nested'/'im.png').write_bytes(b'fake-image-for-mock')
                rows = [{'item_id':str(i),'image_id':'im','input':q,'task_type':'TQA',
                         'dataset_name':'WTQ','output':'SECRET_GOLD'} for i,q in enumerate(['short','long'])]
                (data/'ann.json').write_text(json.dumps(rows))
                cfg = json.loads(Path(__file__).with_name('config.json').read_text())
                cfg.update(data_root=str(data),annotation='ann.json',
                           endpoints=['http://127.0.0.1:8200/v1']*4)
                config = root/'config.json'; config.write_text(json.dumps(cfg))
                args = ['evaluate.py','full','--config',str(config)]
                with patch.object(ev,'ROOT',root), patch.object(sys,'argv',args):
                    ev.main(); ev.main()
                self.assertEqual(len(seen),2)
                self.assertNotIn('SECRET_GOLD',json.dumps(seen))
                summary = json.loads((root/'outputs/full/summary.json').read_text())
                self.assertEqual(summary['truncated'],1)
                self.assertEqual(summary['ok'],2)
                self.assertEqual(len((root/'outputs/full/predictions.jsonl').read_text().splitlines()),2)
                cfg['max_tokens'] = 123; config.write_text(json.dumps(cfg))
                with patch.object(ev,'ROOT',root), patch.object(sys,'argv',args):
                    with self.assertRaisesRegex(ValueError,'changed'):
                        ev.main()
                self.assertFalse((root/'outputs/full/.lock').exists())

    def test_failure_preserved(self):
        with patch.object(ev,'body_for', return_value={}), patch.object(ev,'request',side_effect=ValueError('invalid response')):
            result = ev.infer({'id':'failed'}, {'transport_retries':0,'timeout_seconds':1}, 'http://test')
        self.assertEqual(result['status'],'error')
        self.assertIn('invalid response',result['error'])


if __name__ == '__main__':
    unittest.main()
