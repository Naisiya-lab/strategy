"""Direct image + original question baseline. No strategy framework imports."""
import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def request(url, body=None, timeout=30):
    raw = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=raw, headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def health(cfg):
    for endpoint in cfg['endpoints']:
        data = request(endpoint + '/models')
        assert cfg['served_model'] in [r['id'] for r in data['data']], (endpoint, data)
        print('READY', endpoint, flush=True)


def prepare(cfg, mode, per_group):
    root = Path(cfg['data_root'])
    annotation = root / cfg['annotation']
    data = annotation.read_bytes()
    rows = json.loads(data)
    ids = [r['item_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate item_id in annotations')
    if mode == 'smoke':
        groups = defaultdict(list)
        for row in rows:
            groups[(row['task_type'], row['dataset_name'])].append(row)
        rng = random.Random(cfg['seed'])
        rows = [r for key in sorted(groups) for r in rng.sample(
            groups[key], min(per_group, len(groups[key])))]
    # Build image index once; detect ambiguity instead of guessing an image.
    index = defaultdict(list)
    for p in (root / 'images').rglob('*'):
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
            index[p.stem].append(p)
    hashes = {}
    prepared = []
    for r in rows:
        iid = str(r['image_id'])
        # .csv/.html are part of the table ID, not image extensions.
        image_id = Path(iid)
        key = image_id.stem if image_id.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'} else image_id.name
        matches = index.get(key, [])
        if len(matches) != 1:
            raise ValueError(f'Image {iid}: expected exactly one file, found {len(matches)}')
        p = matches[0]
        if p not in hashes:
            hashes[p] = digest(p.read_bytes())
        if not isinstance(r['input'], str) or not r['input'].strip():
            raise ValueError('Missing question: ' + r['item_id'])
        prepared.append({'id': r['item_id'], 'image_id': iid,
                         'image': str(p), 'image_sha256': hashes[p],
                         'question': r['input'], 'task_type': r['task_type'],
                         'dataset_name': r['dataset_name']})
    return prepared, digest(data)


def body_for(row, cfg):
    p = Path(row['image'])
    data = p.read_bytes()
    if digest(data) != row['image_sha256']:
        raise ValueError('Image changed after preflight: ' + str(p))
    mime = mimetypes.guess_type(p.name)[0] or 'image/jpeg'
    return {'model': cfg['served_model'], 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url':
            f'data:{mime};base64,' + base64.b64encode(data).decode()}},
        {'type': 'text', 'text': row['question']}]}],
        'temperature': cfg['temperature'], 'seed': cfg['seed'],
        'max_tokens': cfg['max_tokens'], 'stream': False}


def infer(row, cfg, endpoint):
    started = time.monotonic()
    attempts = []
    record = dict(row, endpoint=endpoint, status='error')
    try:
        body = body_for(row, cfg)
        for i in range(cfg['transport_retries'] + 1):
            then = time.monotonic()
            try:
                raw = request(endpoint + '/chat/completions', body, cfg['timeout_seconds'])
                attempts.append({'attempt': i + 1, 'seconds': time.monotonic()-then, 'ok': True})
                record['raw_response'] = raw
                choice = raw['choices'][0]
                answer = choice['message'].get('content')
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError('Empty or non-text response; preserved, not silently regenerated')
                record.update(status='ok', answer=answer, finish_reason=choice.get('finish_reason'),
                              usage=raw.get('usage', {}))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors='replace')
                attempts.append({'attempt': i+1, 'seconds': time.monotonic()-then,
                                 'http_status': exc.code, 'error': detail})
                if exc.code not in (429, 500, 502, 503, 504) or i == cfg['transport_retries']:
                    raise RuntimeError(f'HTTP {exc.code}: {detail}') from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                attempts.append({'attempt': i+1, 'seconds': time.monotonic()-then, 'error': str(exc)})
                if i == cfg['transport_retries']:
                    raise
            time.sleep(min(2 ** i, 8))
    except Exception as exc:
        record['error'] = str(exc)
    record.update(seconds=time.monotonic()-started, attempts=attempts)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['health', 'check', 'smoke', 'full'])
    ap.add_argument('--config', type=Path, default=ROOT/'config.json')
    ap.add_argument('--run-name')
    ap.add_argument('--per-group', type=int, default=4)
    ap.add_argument('--retry-errors', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    assert cfg['concurrency_per_endpoint'] > 0 and args.per_group > 0
    if args.mode == 'health':
        health(cfg)
        return
    rows, source_hash = prepare(cfg, args.mode, args.per_group)
    print('Preflight OK:', len(rows), 'samples;', dict(Counter(r['task_type'] for r in rows)), flush=True)
    if args.mode == 'check':
        return
    health(cfg)
    name = args.run_name or args.mode
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('run-name must be a directory name, not a path')
    out = ROOT/'outputs'/name
    out.mkdir(parents=True, exist_ok=True)
    lock = out/'.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Run already locked: {lock}. After a crash, verify no evaluator is running before removing it.')
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        manifest = {'config': cfg, 'annotation_sha256': source_hash, 'mode': args.mode,
                    'rows': rows, 'script_sha256': digest(Path(__file__).read_bytes())}
        manifest_file = out/'manifest.json'
        if manifest_file.exists() and json.loads(manifest_file.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Run inputs/config/code changed; use a new --run-name')
        save(manifest_file, manifest)
        def target(row):
            return out/'samples'/(digest(row['id'].encode())+'.json')
        pending = []
        for row in rows:
            p = target(row)
            old = json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
            if old is None or (args.retry_errors and old['status'] != 'ok'):
                if old is not None:
                    archive = out/'previous_attempts'/(p.stem + '_' + str(time.time_ns()) + '.json')
                    save(archive, old)
                pending.append(row)
        started = time.monotonic()
        # One bounded pool per endpoint guarantees balanced concurrency.
        pools = [ThreadPoolExecutor(max_workers=cfg['concurrency_per_endpoint']) for _ in cfg['endpoints']]
        futures = {}
        try:
            for i, row in enumerate(pending):
                k = i % len(pools)
                futures[pools[k].submit(infer, row, cfg, cfg['endpoints'][k])] = row
            for count, future in enumerate(as_completed(futures), 1):
                record = future.result()
                save(target(futures[future]), record)
                if count % 10 == 0 or count == len(pending):
                    elapsed = time.monotonic()-started
                    print(f'{count}/{len(pending)} newly finished | {60*count/max(elapsed,0.01):.2f} samples/min', flush=True)
        finally:
            for pool in pools:
                pool.shutdown(wait=True)
        records = [json.loads(target(r).read_text(encoding='utf-8')) for r in rows]
        # Include failures as empty answers; retain every selected ID in exports.
        with (out/'predictions.jsonl').open('w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps({'question_id': r['id'], 'text': r.get('answer', ''),
                                    'status': r['status']}, ensure_ascii=False)+'\n')
        summary = {'total': len(records), 'ok': sum(r['status']=='ok' for r in records),
                   'errors': sum(r['status']!='ok' for r in records),
                   'truncated': sum(r.get('finish_reason')=='length' for r in records),
                   'newly_attempted': len(pending), 'session_seconds': time.monotonic()-started,
                   'session_samples_per_minute': 60*len(pending)/max(time.monotonic()-started,0.01),
                   'usage': {k: sum(r.get('usage', {}).get(k, 0) or 0 for r in records)
                             for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
                   'note': 'Runtime summary, not accuracy. Failed IDs remain in predictions. Timed-out calls may have consumed unreported tokens.',
                   'finished_at': datetime.now(timezone.utc).isoformat()}
        save(out/'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
