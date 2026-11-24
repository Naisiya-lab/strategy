import csv
from pathlib import Path
import tempfile
import unittest
from report import ALL_COLUMNS, QA_COLUMNS, STRUCTURE_COLUMNS, values, write_report


class ReportTest(unittest.TestCase):
    def test_screenshot_order_and_metric_units(self):
        self.assertEqual([c[1] for c in QA_COLUMNS],['TABMWP','WTQ','HiTab','TAT-QA','FeTaQA',
            'AIT-QA','TabMCQ','TabFact','InfoTabs','PubHealthTab','ToTTo','HiTab_T2T','Rotowire','WikiBIO'])
        self.assertEqual([c[1] for c in STRUCTURE_COLUMNS],['TSD','TSD','TSD_OOD','TSD_OOD',
            'TCE','TCE_OOD','TCL','TCL_OOD','MCD','RCE','RCE','RCE_OOD','RCE_OOD','TR','TR','TR'])
        reports=[{'task':'TQA','benchmark':'TABMWP','metrics':{'Accuracy':0.6936}},
                 {'task':'TQA','benchmark':'FeTaQA','metrics':{'BLEU score':3.964216}},
                 {'task':'TQA','benchmark':'WTQ','metrics':{'Accuracy':0.0}}]
        vals=values(reports,QA_COLUMNS)
        self.assertEqual(vals[0],'69.36%');self.assertEqual(vals[1],'0.00%')
        self.assertEqual(vals[4],'3.964216');self.assertEqual(vals[10],'—')

    def test_tsv_has_30_ordered_metrics_and_errors_are_not_scores(self):
        records={}
        for c in ALL_COLUMNS:
            r=records.setdefault((c[3],c[4]),{'task':c[3],'benchmark':c[4],'metrics':{},
                 'n':1,'execution_errors':0,'truncated':0,'scoring_error':None})
            r['metrics'][c[5]]=1.0
        records[('T2T','ToTTo')]['scoring_error']='Reference missing'
        with tempfile.TemporaryDirectory() as tmp:
            write_report(list(reversed(list(records.values()))),tmp)
            with (Path(tmp)/'SUMMARY.tsv').open(encoding='utf-8-sig') as f:
                rows=list(csv.reader(f,delimiter='\t'))
            self.assertEqual(len(rows),2);self.assertEqual(len(rows[0]),31)
            self.assertEqual(rows[1][11],'—')
            self.assertEqual(rows[0][-3:],['TR HTML TEDS','TR Markdown TEDS','TR LaTeX TEDS'])
            self.assertTrue((Path(tmp)/'METRICS.md').exists())


if __name__=='__main__':unittest.main()
