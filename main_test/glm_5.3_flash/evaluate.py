"""Standalone GLM direct baseline; no strategy or reference answers in requests."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import mimetypes
import os
from pathlib import Path
import random
import threading
import time
import urllib.request
import urllib.error
from tqdm import tqdm
from data import prepare, digest, save

ROOT = Path(__file__).resolve().parent
LEGACY_PROGRESS_HASHES = ['a3b1fce2412eaf43c1aadf2189f36b4542741512b6ef6647e3fdf5db67ffa106', 'f25a283a549f88d7976ee9522df1ce28674b45f793489ae9e4cd55e36767944a']

class Gate:
    def __init__(self):
        self.lock = threading.Lock()
        self.until = 0
        self.stop = threading.Event()
    def pause(self, seconds):
        with self.lock:
            self.until = max(self.until, time.monotonic() + seconds)
    def enter(self):
        while not self.stop.is_set():
            with self.lock:
                delay = self.until - time.monotonic()
            if delay <= 0: return True
            self.stop.wait(min(delay, 1))
        return False

def retry_delay(value, attempt):
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return min(60, 2 ** (attempt + 1)) + random.random()

def body_for(row, cfg):
    p = Path(row['image']); data = p.read_bytes()
    if digest(data) != row['image_sha256']: raise ValueError('Image changed: '+str(p))
    mime = mimetypes.guess_type(p.name)[0] or 'image/jpeg'
    return {'model': cfg['model_id'], 'stream': False, 'max_tokens': cfg['max_tokens'],
            **cfg['request_options'], 'messages': [{'role':'user','content':[
                {'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(data).decode()}},
                {'type':'text','text':row['question']}]}]}

def infer(row, cfg, key, gate):
    record = dict(row, status='error', attempts=[])
    started = time.monotonic()
    try:
        body = json.dumps(body_for(row,cfg)).encode()
        for attempt in range(cfg['transport_retries']+1):
            if not gate.enter(): raise RuntimeError('Stopped before request')
            then = time.monotonic()
            try:
                req = urllib.request.Request(cfg['endpoint'], data=body, headers={
                    'Content-Type':'application/json','Authorization':'Bearer '+key})
                with urllib.request.urlopen(req,timeout=cfg['timeout_seconds']) as response:
                    raw = json.load(response)
                record['raw_response'] = raw
                record['usage'] = raw.get('usage', {})
                choice = raw['choices'][0]; answer = choice['message'].get('content')
                record['finish_reason'] = choice.get('finish_reason')
                record['attempts'].append({'attempt':attempt+1,'seconds':time.monotonic()-then,'ok':True})
                if not isinstance(answer,str) or not answer.strip(): raise ValueError('Empty/non-text answer')
                record.update(status='ok',answer=answer)
                break
            except urllib.error.HTTPError as exc:
                # Do not persist service error bodies which may echo request credentials.
                record['attempts'].append({'attempt':attempt+1,'http_status':exc.code,'seconds':time.monotonic()-then})
                if exc.code not in (429,500,502,503,504):
                    gate.stop.set()
                    raise RuntimeError(f'HTTP {exc.code}; batch stopped, check credentials/quota/request') from None
                delay = retry_delay(exc.headers.get('Retry-After'), attempt)
                gate.pause(delay)
                if attempt == cfg['transport_retries']: raise RuntimeError(f'HTTP {exc.code}; retries exhausted') from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                record['attempts'].append({'attempt':attempt+1,'error_type':type(exc).__name__,'seconds':time.monotonic()-then})
                if attempt == cfg['transport_retries']: raise RuntimeError('Network retries exhausted') from None
                gate.stop.wait(retry_delay(None,attempt))
    except Exception as exc:
        # Exception repr can escape control characters and expose credentials.
        # Never persist arbitrary exception text from request construction.
        record['error'] = 'Request failed: ' + type(exc).__name__
        if isinstance(exc, RuntimeError) and str(exc).startswith(('HTTP ', 'Network retries', 'Stopped before')):
            record['error'] = str(exc)
    record['seconds'] = time.monotonic()-started
    return record

def export(out, rows):
    records=[]
    for row in rows:
        p=out/'samples'/(digest(row['id'].encode())+'.json')
        if p.exists(): records.append(json.loads(p.read_text(encoding='utf8')))
    summary={'total':len(rows),'saved':len(records),'pending':len(rows)-len(records),
             'ok':sum(r['status']=='ok' for r in records),'errors':sum(r['status']!='ok' for r in records),
             'truncated':sum(r.get('finish_reason')=='length' for r in records),
             'usage':{k:sum(r.get('usage',{}).get(k,0) or 0 for r in records) for k in ('prompt_tokens','completion_tokens','total_tokens')},
             'updated_at':datetime.now(timezone.utc).isoformat(),
             'note':'Usage may exclude unsuccessful or timed-out billable requests. Truncated text is retained.'}
    save(out/'summary.json',summary)
    tmp=out/'predictions.jsonl.tmp'
    with tmp.open('w',encoding='utf8') as f:
        for r in records: f.write(json.dumps({'question_id':r['id'],'text':r.get('answer',''),'status':r['status']},ensure_ascii=False)+'\n')
    tmp.replace(out/'predictions.jsonl')
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['check','smoke','full'])
    ap.add_argument('--config',type=Path,default=ROOT/'config.json')
    ap.add_argument('--concurrency',type=int)
    ap.add_argument('--run-name')
    ap.add_argument('--retry-errors',action='store_true')
    args=ap.parse_args();cfg=json.loads(args.config.read_text(encoding='utf8'))
    workers=cfg['concurrency'] if args.concurrency is None else args.concurrency
    if not 1 <= workers <= 50: ap.error('Concurrency must be 1..50')
    name=args.run_name or args.mode
    if Path(name).name!=name or name in ('.','..'): ap.error('Invalid run name')
    key=os.environ.get(cfg['api_key_env'],'')
    if args.mode!='check' and not key: ap.error('Set '+cfg['api_key_env']+' first')
    if args.mode!='check' and (not key.isascii() or any(c.isspace() or ord(c)<33 or ord(c)==127 for c in key)):
        ap.error('Invalid API key: whitespace/control/non-ASCII characters. Re-enter the key only.')
    if args.mode=='check':
        rows,_=prepare(cfg,'full',4);print('Validated',len(rows),'samples');return 0
    out=ROOT/'outputs'/name;out.mkdir(parents=True,exist_ok=True)
    lock=out/'.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    gate=Gate();rows=[];code=0
    try:
        rows,source_hash=prepare(cfg,args.mode,4)
        semantic={k:v for k,v in cfg.items() if k not in ('concurrency','timeout_seconds','transport_retries')}
        manifest={'config':semantic,'annotation_sha256':source_hash,'mode':args.mode,'rows':rows,
                  'script_sha256':digest(Path(__file__).read_bytes()),'data_script_sha256':digest((ROOT/'data.py').read_bytes())}
        mf=out/'manifest.json'
        if mf.exists():
            previous=json.loads(mf.read_text(encoding='utf8'))
            comparable=dict(previous)
            # Only the verified pre-tqdm runner is compatible; keep original provenance.
            if comparable.get('script_sha256') in LEGACY_PROGRESS_HASHES:
                comparable['script_sha256']=manifest['script_sha256']
            if comparable != manifest: raise ValueError('Inputs/code changed; use new --run-name')
        else:
            save(mf,manifest)
        save(out/'session.json',{'script_sha256':digest(Path(__file__).read_bytes()),'concurrency':workers,'timeout_seconds':cfg['timeout_seconds'],'transport_retries':cfg['transport_retries']})
        def target(row): return out/'samples'/(digest(row['id'].encode())+'.json')
        pending=[]
        for row in rows:
            p=target(row);old=json.loads(p.read_text(encoding='utf8')) if p.exists() else None
            if old is None or (args.retry_errors and old['status']!='ok'): pending.append(row)
        print(f'{len(rows)} selected; {len(pending)} pending; concurrency={workers}',flush=True)
        def process(row):
            p=target(row)
            if p.exists(): save(out/'previous_attempts'/(p.stem+'_'+str(time.time_ns())+'.json'),json.loads(p.read_text(encoding='utf8')))
            record=infer(row,cfg,key,gate);save(p,record);return record
        pool=ThreadPoolExecutor(max_workers=workers);futures=set();iterator=iter(pending);done=0
        progress=tqdm(total=len(pending), desc='GLM inference', unit='sample',
                      ascii=True, dynamic_ncols=True, mininterval=0.5,
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        successes=errors=0
        def submit():
            row=next(iterator,None)
            if row is not None: futures.add(pool.submit(process,row))
        try:
            for _ in range(min(workers,len(pending))): submit()
            while futures:
                finished,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
                if not finished: progress.refresh()
                for future in finished:
                    futures.remove(future);r=future.result();done+=1
                    successes += r['status']=='ok'
                    errors += r['status']!='ok'
                    progress.set_postfix(ok=successes, errors=errors, workers=workers, refresh=False)
                    progress.update(1)
                    if not gate.stop.is_set(): submit()
        except KeyboardInterrupt:
            code=130;gate.stop.set();tqdm.write('Stopping; saving in-flight responses...')
        finally:
            pool.shutdown(wait=True)
            progress.close()
        summary=export(out,rows);print(json.dumps(summary,indent=2))
        return code or (1 if summary['errors'] or summary['pending'] else 0)
    finally: lock.unlink(missing_ok=True)

if __name__=='__main__': raise SystemExit(main())
