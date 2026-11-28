import json, tempfile, unittest
from pathlib import Path
from baseline_io import prepare

class ImageIds(unittest.TestCase):
 def test_compound_and_image_extensions(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); (root/'images/nested').mkdir(parents=True)
   ids=['TabFact_2-10689762-4.html.csv', 'ordinary', 'with_extension.png']
   filenames=['TabFact_2-10689762-4.html.csv.jpg','ordinary.jpg','with_extension.png']
   rows=[]
   for i,(iid,filename) in enumerate(zip(ids,filenames)):
    (root/'images/nested'/filename).write_bytes(b'image fixture')
    rows.append(dict(item_id=str(i),image_id=iid,input='Question?',task_type='TQA',dataset_name='WTQ'))
   (root/'ann.json').write_text(json.dumps(rows))
   cfg=dict(data_root=d,annotation='ann.json',seed=42)
   prepared,_=prepare(cfg,'check',1)
   self.assertEqual([Path(r['image']).name for r in prepared],filenames)
   (root/'images'/filenames[0]).write_bytes(b'duplicate')
   with self.assertRaisesRegex(ValueError,'found 2'): prepare(cfg,'check',1)
if __name__=='__main__': unittest.main()
