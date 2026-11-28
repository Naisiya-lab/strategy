import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('qwen_score',Path(__file__).with_name('score.py'))
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)


class ScoreTest(unittest.TestCase):
    def test_teds_format_denominators(self):
        kinds={'a':'HTML','b':'HTML','c':'Markdown','d':'Latex'}
        gold={k:{'original_query_type':'table recognition with '+v} for k,v in kinds.items()}
        results={k:{'score':1.,'error':None} for k in kinds}
        results['b']={'score':0.,'error':'No HTML table'}
        metrics, groups=score.aggregate_teds([{'question_id':k} for k in kinds],gold,results)
        self.assertEqual(metrics,{'HTML_TEDS':0.5,'Markdown_TEDS':1.,'LaTeX_TEDS':1.})
        self.assertEqual(groups['HTML']['n'],2)
        self.assertEqual(groups['HTML']['parse_or_compute_errors'],1)

    def test_source_hash_accepts_line_endings_rejects_edits(self):
        original=(score.OFFICIAL/'MMTab_evaluation.ipynb').read_bytes().replace(b'\r\n',b'\n')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'MMTab_evaluation.ipynb'
            with patch.object(score,'OFFICIAL',root):
                try:
                    for content in (original, original.replace(b'\n',b'\r\n')):
                        p.write_bytes(content);score.functions.cache_clear()
                        self.assertIn('evaluate_mcd_questions',score.functions())
                    p.write_bytes(original+b' ');score.functions.cache_clear()
                    with self.assertRaisesRegex(ValueError,'hash changed'):
                        score.functions()
                finally:
                    score.functions.cache_clear()

    def test_qa_failure_keeps_denominator(self):
        gold={str(i):{'item_id':str(i),'answer_list':['42']} for i in range(2)}
        result=score.official_group('TQA','WTQ',[
            {'question_id':'0','text':'{"answer": ["42"]}'},
            {'question_id':'1','text':''}],gold,{})
        self.assertIsNone(result['scoring_error'])
        self.assertEqual(result['metrics']['Accuracy'],0.5)

    def test_teds_exact_and_empty(self):
        item={'original_query_type':'table recognition with HTML',
              'output':'<table><tr><td>42</td></tr></table>'}
        _,good=score.teds_one(('a',item,item['output']))
        _,empty=score.teds_one(('b',item,''))
        self.assertEqual(good['score'],1.0)
        self.assertEqual(empty['score'],0.0)

    def test_manifest_and_complete_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data';data.mkdir();run=root/'run';(run/'samples').mkdir(parents=True)
            gold=[{'item_id':'a','answer_list':['42'],'image_id':'image'}]
            ann=data/'ann.json';ann.write_text(json.dumps(gold))
            (data/'MMTab-eval_test_tables_23K.json').write_text('[]')
            manifest={'config':{'data_root':str(data),'annotation':'ann.json'},
                      'mode':'full','annotation_sha256':score.sha(ann.read_bytes()),
                      'rows':[{'id':'a','task_type':'TQA','dataset_name':'WTQ'}]}
            (run/'manifest.json').write_text(json.dumps(manifest))
            args=['score.py','--run',str(run)]
            with patch.object(sys,'argv',args):
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    score.main()
                (run/'samples'/(score.sha(b'a')+'.json')).write_text(json.dumps({
                    'id':'a','status':'ok','answer':'{"answer": ["42"]}','finish_reason':'length'}))
                score.main()
            result=json.loads((run/'metrics_normalized/metrics.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(result['groups'][0]['metrics']['Accuracy'],1.0)
            self.assertEqual(result['groups'][0]['truncated'],1)


if __name__=='__main__':unittest.main()
