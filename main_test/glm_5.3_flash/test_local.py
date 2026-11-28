import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import evaluate as e

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);p=self.root/'x.png';p.write_bytes(b'image')
        self.row={'id':'one','image':str(p),'image_sha256':e.digest(b'image'),'question':'original question','task_type':'TQA','dataset_name':'WTQ'}
        self.cfg=json.loads((e.ROOT/'config.json').read_text())
    def raw(self):
        return io.BytesIO(json.dumps({'choices':[{'message':{'content':'42'},'finish_reason':'stop'}],'usage':{'total_tokens':9}}).encode())
    def test_request_parity(self):
        b=e.body_for(self.row,self.cfg)
        self.assertEqual(b['reasoning_effort'],'high');self.assertEqual(b['max_tokens'],8192)
        self.assertEqual(b['messages'][0]['content'][1]['text'],'original question')
        self.assertNotIn('temperature',b);self.assertNotIn('seed',b)
    def test_rate_limit_then_success(self):
        error=urllib.error.HTTPError('url',429,'limited',{'Retry-After':'0'},None)
        with patch.object(e.urllib.request,'urlopen',side_effect=[error,self.raw()]):
            result=e.infer(self.row,self.cfg,'secret',e.Gate())
        self.assertEqual(result['status'],'ok');self.assertEqual(len(result['attempts']),2)
    def test_fatal_stops_batch(self):
        gate=e.Gate()
        with patch.object(e.urllib.request,'urlopen',side_effect=urllib.error.HTTPError('url',401,'secret',{},None)):
            result=e.infer(self.row,self.cfg,'secret',gate)
        self.assertTrue(gate.stop.is_set());self.assertNotIn('secret',str(result))
    def test_resume_and_concurrency_change(self):
        (self.root/'config.json').write_text(json.dumps(self.cfg))
        (self.root/'data.py').write_text('fixture')
        args=['evaluate.py','full','--config',str(self.root/'config.json')]
        with patch.object(e,'ROOT',self.root),patch.object(e,'prepare',return_value=([self.row],'hash')),patch.dict(e.os.environ,{'ZHIPU_API_KEY':'secret'}):
            with patch('sys.argv',args),patch.object(e.urllib.request,'urlopen',return_value=self.raw()) as api:
                self.assertEqual(e.main(),0);self.assertEqual(api.call_count,1)
            with patch('sys.argv',args+['--concurrency','30']),patch.object(e.urllib.request,'urlopen') as api:
                self.assertEqual(e.main(),0);api.assert_not_called()
            manifest=json.loads((self.root/'outputs/full/manifest.json').read_text())
            self.assertNotIn('secret',str(manifest));self.assertEqual(manifest['rows'][0]['id'],'one')
    def test_empty_response_is_error(self):
        raw=io.BytesIO(b'{"choices":[{"message":{"content":""},"finish_reason":"length"}]}')
        with patch.object(e.urllib.request,'urlopen',return_value=raw):r=e.infer(self.row,self.cfg,'secret',e.Gate())
        self.assertEqual(r['status'],'error');self.assertEqual(r['finish_reason'],'length')

if __name__=='__main__':unittest.main()
