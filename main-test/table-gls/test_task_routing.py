import unittest
import evaluate as e
class Routing(unittest.TestCase):
 def test_non_qa_uses_complete_instruction(self):
  for task in ['TR','T2T','TFV','TSD','TCL','RCE','MCD','TCE','OOD_TSD']:
   row=dict(task_type=task,dataset_name='HiTab',question='Convert whole table into HTML.',original_query='')
   for stage in (1,2,3):
    p=e.prompt(stage,row,'{"x":1}',e.templates())
    self.assertIn('Convert whole table into HTML.',p)
    self.assertIn('Task:',p)
if __name__=='__main__': unittest.main()
