import json,tempfile,unittest
from pathlib import Path
from shard_runner import run_shards,shard_rows

def fake_worker(i,gpu,cfg,rows,out,log,events):
 events.put((i,"ready",None))
 for row in rows:
  with (Path(out)/(row['id']+'.json')).open('x') as f:json.dump({'gpu':gpu,'shard':i},f)
  events.put((i,"sample",row['id']))
 events.put((i,"done",None))

def failed_worker(*args):
 raise RuntimeError('simulated worker failure')

class Shards(unittest.TestCase):
 def test_partition_resume_and_parallel_completion(self):
  rows=[{'id':str(i)} for i in range(7)]
  parts=shard_rows(rows,2)
  self.assertEqual([len(p) for p in parts],[4,3])
  self.assertFalse({r['id'] for r in parts[0]} & {r['id'] for r in parts[1]})
  with tempfile.TemporaryDirectory() as d:
   cfg=dict(gpus='0',gpu_groups=['0','1'],tensor_parallel_size=1)
   run_shards(cfg,rows,rows[2:],Path(d),Path(d),'test',fake_worker)
   self.assertEqual(sorted(p.stem for p in Path(d).glob('*.json')),['2','3','4','5','6'])
   for p in Path(d).glob('*.json'):self.assertEqual(json.loads(p.read_text())['gpu'],str(int(p.stem)%2))
 def test_worker_failure_propagates(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(RuntimeError,'Shard 0 exited'):
    run_shards(dict(gpus='0',tensor_parallel_size=1),[{'id':'0'}],[{'id':'0'}],Path(d),Path(d),'test',failed_worker)
if __name__=='__main__':unittest.main()
