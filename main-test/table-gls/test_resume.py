import tempfile, unittest
from pathlib import Path
import evaluate as e

class ResumeTests(unittest.TestCase):
 def test_resume_after_stage_two_failure(self):
  row={"task_type":"TQA","id":"a","dataset_name":"WTQ","original_query":"Which?","question":"Which?"}
  ts=e.templates(); calls=[]
  def broken(text,row,stage):
   calls.append(stage)
   if stage==2: raise RuntimeError("simulated interruption")
   return '{"target_columns": []}',"stop",{}
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)
   with self.assertRaises(RuntimeError): e.process(row,{},out,broken,ts)
   self.assertEqual(calls,[1,2])
   calls.clear()
   def good(text,row,stage):
    calls.append(stage)
    return ('Sub-table: x' if stage==2 else 'Reasoning: x\n{"answer":"7"}'),"stop",{}
   result=e.process(row,{},out,good,ts)
   self.assertEqual(calls,[2,3])
   self.assertEqual(result['status'],'ok')
   calls.clear(); e.process(row,{},out,good,ts)
   self.assertEqual(calls,[])
 def test_official_prompt_parity(self):
  import ast,json
  class FakeImage:
   @staticmethod
   def open(path): return FakeImage()
   def convert(self,mode): return self
  ts=e.templates()
  with tempfile.TemporaryDirectory() as d:
   previous=Path(d)/'previous.jsonl'
   for ds in ['AIT-QA','TabMCQ','WTQ','HiTab','TAT-QA','TabFact','TabMWP','FeTaQA']:
    row={'task_type':'TQA','id':'a','image_id':'image','question':'Task e.g., x','original_query':'Question?','dataset_name':ds}
    sample=dict(row,item_id='a',input=row['question'])
    for stage in (1,2,3):
     prior='```json\n{"thought":"x"}\n```' if stage==2 else 'Plan Evaluation: ok\nSub-table: x'
     previous.write_text(json.dumps({'item_id':'a','prediction':prior})+'\n')
     source=e.ROOT/'official'/f'eval_mmtab_vllm_tablegls_stage{stage}.py'
     tree=ast.parse(source.read_text())
     nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_prompt_qwen3_vl']
     env=dict(ts[stage],os=__import__('os'),json=json,Image=FakeImage,tqdm=lambda x:x)
     exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
     args=([sample],d,'qwen3vl') if stage==1 else ([sample],d,str(previous),'qwen3vl')
     expected=env['create_prompt_qwen3_vl'](*args)[0]['prompt']
     self.assertEqual(e.prompt(stage,row,prior,ts),expected)
if __name__=='__main__': unittest.main()
