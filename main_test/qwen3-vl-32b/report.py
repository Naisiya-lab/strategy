"""Fixed paper-table presentation; never changes metric computation."""
import csv
from pathlib import Path

# section, display label, metric label, task, stored benchmark, metric key
QA_COLUMNS = [
    ('TQA','TABMWP','Acc','TQA','TABMWP','Accuracy'),
    ('TQA','WTQ','Acc','TQA','WTQ','Accuracy'),
    ('TQA','HiTab','Acc','TQA','HiTab','Accuracy'),
    ('TQA','TAT-QA','Acc','TQA','TAT-QA','Accuracy'),
    ('TQA','FeTaQA','BLEU','TQA','FeTaQA','BLEU score'),
    ('TQA','AIT-QA','Acc','TQA','AIT-QA','Accuracy'),
    ('TQA','TabMCQ','Acc','TQA','TabMCQ','Accuracy'),
    ('TFV','TabFact','Acc','TFV','TabFact','Accuracy'),
    ('TFV','InfoTabs','Acc','TFV','InfoTabs','Accuracy'),
    ('TFV','PubHealthTab','Acc','TFV','PubHealthTab','Accuracy'),
    ('T2T','ToTTo','BLEU','T2T','ToTTo','BLEU score'),
    ('T2T','HiTab_T2T','BLEU','T2T','HiTab_t2t','BLEU score'),
    ('T2T','Rotowire','BLEU','T2T','Rotowire','BLEU score'),
    ('T2T','WikiBIO','BLEU','T2T','WikiBIO','BLEU score'),
]
STRUCTURE_COLUMNS = [
    ('TSU','TSD','Row ACC','TSD','TSD','row number accuracy'),
    ('TSU','TSD','Col ACC','TSD','TSD','column number accuracy'),
    ('TSU','TSD_OOD','Row ACC','OOD_TSD','OOD_TSD','row number accuracy'),
    ('TSU','TSD_OOD','Col ACC','OOD_TSD','OOD_TSD','column number accuracy'),
    ('TSU','TCE','Acc','TCE','TCE','cell level accuracy'),
    ('TSU','TCE_OOD','Acc','OOD_TCE','OOD_TCE','cell level accuracy'),
    ('TSU','TCL','Acc','TCL','TCL','Cell-level accuracy'),
    ('TSU','TCL_OOD','Acc','OOD_TCL','OOD_TCL','Cell-level accuracy'),
    ('TSU','MCD','F1','MCD','MCD','F1 score'),
    ('TSU','RCE','Row F1','RCE','RCE','Row-level F1'),
    ('TSU','RCE','Col F1','RCE','RCE','Column-level F1'),
    ('TSU','RCE_OOD','Row F1','OOD_RCE','OOD_RCE','Row-level F1'),
    ('TSU','RCE_OOD','Col F1','OOD_RCE','OOD_RCE','Column-level F1'),
    ('TR','TR','HTML TEDS','TR','TR','HTML_TEDS'),
    ('TR','TR','Markdown TEDS','TR','TR','Markdown_TEDS'),
    ('TR','TR','LaTeX TEDS','TR','TR','LaTeX_TEDS'),
]
ALL_COLUMNS=QA_COLUMNS+STRUCTURE_COLUMNS


def group_order(key):
    return next((i for i,c in enumerate(ALL_COLUMNS) if (c[3],c[4])==key),len(ALL_COLUMNS))


def values(reports, columns):
    lookup={(r['task'],r['benchmark']):r for r in reports}
    output=[]
    for section,label,metric,task,benchmark,key in columns:
        r=lookup.get((task,benchmark))
        if r is None or r.get('scoring_error') or key not in r['metrics']:
            output.append('—')
        else:
            value=r['metrics'][key]
            output.append(f'{value:.6f}' if metric=='BLEU' else f'{100*value:.2f}%')
    return output


def write_report(reports, out, model='Qwen3-VL-32B-Instruct'):
    out=Path(out)
    protocol = 'V2规范化评分' if out.name == 'metrics_normalized' else '原始官方评分'
    lines=['# '+model+' MMTab-eval指标','', protocol+'。预测和参考对称处理；BLEU保留大小写，TR沿用原始三格式TEDS。','',
           '顺序与指定论文表格一致。Acc、F1、TEDS显示百分数；BLEU保持0–100，保留6位小数。',
           '“—”表示未评测、指标不可用或评分异常，不代表0。ToTTo缺少测试参考答案时不本地评分。','']
    for title,columns,filename in [('TQA / TFV / T2T',QA_COLUMNS,'TQA_TFV_T2T.tsv'),
                                   ('TSU / TR',STRUCTURE_COLUMNS,'TSU_TR.tsv')]:
        headers=[c[1]+' '+c[2] for c in columns]
        vals=values(reports,columns)
        lines += ['## '+title,'','| 模型 | '+' | '.join(headers)+' |',
                  '|---|'+'|'.join(['---:']*len(columns))+'|',
                  '| '+model+' | '+' | '.join(vals)+' |','']
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['模型']+headers);writer.writerow([model]+vals)
    with (out/'SUMMARY.tsv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,delimiter='\t')
        writer.writerow(['模型']+[c[1]+' '+c[2] for c in ALL_COLUMNS])
        writer.writerow([model]+values(reports,ALL_COLUMNS))
    lines += ['## 分项审计','',
              'TR每种格式以全部对应样本为分母，转换/计算失败记0。RCE等其他任务保留官方公式及其解析局限。',
              'metrics.json保留原始数值、官方日志、评分异常和版本信息；不混合计算总准确率。','',
              '| 任务 | 子集 | 样本数 | 执行失败 | 截断 | 评分说明 |',
              '|---|---|---:|---:|---:|---|']
    for r in sorted(reports,key=lambda r:group_order((r['task'],r['benchmark']))):
        note=(r.get('scoring_error') or '').replace('|','/').replace('\n',' ')
        lines.append(f"| {r['task']} | {r['benchmark']} | {r['n']} | {r['execution_errors']} | {r['truncated']} | {note} |")
        for fmt,d in r.get('format_breakdown',{}).items():
            lines.append(f"| TR | {fmt} | {d['n']} | — | — | 转换/计算失败 {d['parse_or_compute_errors']} |")
    (out/'METRICS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
