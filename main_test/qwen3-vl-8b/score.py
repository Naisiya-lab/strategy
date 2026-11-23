"""Offline MMTab notebook metrics. No model calls; no answer rewriting."""
import argparse
import ast
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT/'scoring_source'
# Hash the frozen source after CRLF -> LF normalization. Git normalizes text
# on Windows/Linux; verify content without treating line endings as edits.
NOTEBOOK_HASH = '81b8a5fc48a11fc06d699833c94715b075e015f92b383558f70d1fe41d659aca'


def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)


@lru_cache(maxsize=1)
def functions():
    data = (OFFICIAL/'MMTab_evaluation.ipynb').read_bytes()
    if sha(data.replace(b'\r\n', b'\n')) != NOTEBOOK_HASH:
        raise ValueError('Scoring notebook hash changed')
    nb = json.loads(data)
    nodes = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        for i in (2,10,12):
            nodes.extend(n for n in ast.parse(''.join(nb['cells'][i]['source'])).body
                         if isinstance(n, ast.FunctionDef))
    ns = {'re':re, 'json':json, 'eval':ast.literal_eval}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen_official_functions', 'exec'), ns)
    return ns


def official_group(task, benchmark, rows, gold, tables):
    from sacrebleu.metrics import BLEU
    ns = dict(functions())
    # Functions share their original globals; use a fresh binding for this call.
    captured, log = {}, []
    def capture(*args, **kw):
        log.append(' '.join(map(str,args)))
        label = str(args[0]).strip().rstrip(':') if args else ''
        metric_label = any(w in label.lower() for w in ('accuracy','precision','recall','f1'))
        if len(args)==2 and metric_label and isinstance(args[1], (float,int)):
            value = float(args[1])
            if math.isfinite(value): captured[str(args[0]).strip().rstrip(':')] = value
        elif len(args)==2 and hasattr(args[1], 'score'):
            captured['BLEU score'] = float(args[1].score)
    shared = functions()
    shared.update(print=capture, BLEU=BLEU, item_id_to_test_item=gold, table_id_to_test_table=tables)
    if task in ('TQA','TFV'):
        name = ('evaluate_text_generation_questions' if benchmark=='FeTaQA' else
                'evaluate_tabmcq_questions' if benchmark=='TabMCQ' else 'evaluate_tqa_questions')
    elif task=='T2T': name='evaluate_text_generation_questions'
    else:
        name = {'TSD':'evaluate_tsd_questions','TCE':'evaluate_tce_questions',
                'TCL':'evaluate_tcl_questions','RCE':'evaluate_rce_questions',
                'MCD':'evaluate_mcd_questions'}[task.removeprefix('OOD_')]
    error = None
    try:
        ns[name](benchmark, [dict(r) for r in rows])
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    return {'metrics':captured, 'official_log':log, 'scoring_error':error}


def teds_one(job):
    sid, item, answer = job
    sys.path.insert(0,str(OFFICIAL))
    from metric import TEDS
    ns = functions()
    if not answer.strip(): return sid, {'score':0., 'error':'Empty prediction'}
    try:
        kind = item['original_query_type']; gold = item['output']
        if 'HTML' in kind:
            text = answer.replace('\n','')
            if '<body' in text: text = re.findall('<body.*',text)[0]
            elif '<table' in text: text = re.findall('<table.*',text)[0]
            else: raise ValueError('No HTML table in prediction')
            ph, gh = ns['wrap_html_table'](text), ns['wrap_html_table'](gold)
        elif 'Latex' in kind:
            ph, gh = ns['convert_latex_table_to_html'](answer), ns['convert_latex_table_to_html'](gold)
        else:
            ph, gh = ns['convert_markdown_table_to_html'](answer), ns['convert_markdown_table_to_html'](gold)
        value = TEDS(n_jobs=1).evaluate(ph,gh)
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('No finite TEDS result')
        return sid, {'score':float(value), 'error':None}
    except Exception as exc:
        return sid, {'score':0., 'error':type(exc).__name__+': '+str(exc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,default=ROOT/'outputs/full')
    ap.add_argument('--data-root',type=Path)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    if args.workers<1: ap.error('workers must be positive')
    run=args.run.resolve(); manifest=read(run/'manifest.json')
    data=args.data_root or Path(manifest['config']['data_root'])
    ann=data/manifest['config']['annotation']; raw=ann.read_bytes()
    if sha(raw)!=manifest['annotation_sha256']: raise ValueError('Annotation differs from run')
    gold={r['item_id']:r for r in json.loads(raw)}
    tables={r['image_id']:r for r in read(data/'MMTab-eval_test_tables_23K.json')}
    rows=manifest['rows']; ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate manifest ID')
    if manifest['mode']=='full' and set(ids)!=set(gold): raise ValueError('Full run does not cover annotations')
    predictions={}; states={}; groups=defaultdict(list)
    for r in rows:
        sid=r['id']; p=run/'samples'/(sha(sid.encode())+'.json')
        if not p.exists(): raise ValueError('Incomplete run; missing '+sid)
        result=read(p)
        if result['id']!=sid: raise ValueError('Result ID mismatch')
        predictions[sid]=result.get('answer','') if result['status']=='ok' else ''
        states[sid]={'status':result['status'],'finish_reason':result.get('finish_reason')}
        task=r['task_type']; benchmark=r['dataset_name'] if task in ('TQA','TFV','T2T') else task
        groups[(task,benchmark)].append({'question_id':sid,'text':predictions[sid]})
    source_hashes={p.name:sha(p.read_bytes()) for p in OFFICIAL.iterdir() if p.is_file()}
    provenance={'manifest_sha256':sha((run/'manifest.json').read_bytes()),'sources':source_hashes,
                'script_sha256':sha(Path(__file__).read_bytes()),
                'dependencies':{k:version(k) for k in ('sacrebleu','apted','Distance','lxml','tqdm')}}
    out=run/'metrics'; out.mkdir(exist_ok=True)
    reports=[]
    for (task,benchmark),items in sorted(groups.items()):
        print('Scoring',task,benchmark,len(items),flush=True)
        report={'task':task,'benchmark':benchmark,'n':len(items),
                'execution_errors':sum(states[r['question_id']]['status']!='ok' for r in items),
                'truncated':sum(states[r['question_id']]['finish_reason']=='length' for r in items)}
        if benchmark=='ToTTo':
            report.update(metrics={},scoring_error='ToTTo test references unavailable; official external evaluation required')
        elif task=='TR':
            cache=out/'teds_cache'; cache.mkdir(exist_ok=True)
            results={}; todo=[]; paths={}
            for r in items:
                sid=r['question_id']
                key=sha(json.dumps([gold[sid],r['text'],provenance],sort_keys=True).encode())
                paths[sid]=cache/(key+'.json')
                if paths[sid].exists(): results[sid]=read(paths[sid])
                else: todo.append((sid,gold[sid],r['text']))
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for sid,value in pool.map(teds_one,todo,chunksize=1):
                    results[sid]=value; write(paths[sid],value)
                    if len(results)%100==0: print('TEDS',len(results),'/',len(items),flush=True)
            report['metrics']={'TEDS_all_samples':sum(v['score'] for v in results.values())/len(items)}
            report['teds_parse_or_compute_errors']=sum(bool(v['error']) for v in results.values())
            report['scoring_error']=None
            write(out/'teds_per_sample.json',results)
        else:
            report.update(official_group(task,benchmark,items,gold,tables))
        reports.append(report)
        write(out/'metrics.json',{'provenance':provenance,'groups':reports,'complete':False})
    write(out/'metrics.json',{'provenance':provenance,'groups':reports,'complete':True})
    lines=['# Qwen3-VL-8B MMTab-eval metrics','',
           'Accuracy/F1/TEDS use 0–1; corpus BLEU uses 0–100. No mixed overall accuracy.',
           'TR retains all selected samples in denominator; parse/compute failures score zero.',
           'Other tasks use frozen official notebook formulas, including their parsing/denominator limitations.',
           'RCE official F1 may ignore unparsed targets or raise division by zero; see official_log/scoring_error in JSON.',
           '', '| Task | Subset | N | Execution errors | Truncated | Metrics | Scoring note |',
           '|---|---|---:|---:|---:|---|---|']
    for r in reports:
        values='; '.join(f'{k}={v:.6f}' for k,v in r['metrics'].items()) or '—'
        lines.append(f"| {r['task']} | {r['benchmark']} | {r['n']} | {r['execution_errors']} | {r['truncated']} | {values} | {r.get('scoring_error') or ''} |")
    (out/'METRICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved:',out/'METRICS.md',flush=True)


if __name__=='__main__': main()
