"""TabFlash official model/processing, with durable per-item records and resume."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from importlib.metadata import version, PackageNotFoundError
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import uuid

from data import ROOT, digest, prepare


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8', newline='\n') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name != 'nt':
            fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def save(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


@contextmanager
def run_lock(out):
    # OS releases this lock on crashes, including SIGKILL. Never unlink it:
    # deleting a locked inode can let two processes acquire different locks.
    with (out / '.lock').open('a+b') as f:
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            if os.fstat(f.fileno()).st_size == 0:
                f.write(b'0'); f.flush()
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Run is already active; use another run-name') from exc
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Run is already active; use another run-name') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def sample_path(out, sid):
    return Path(out) / 'samples' / (digest(sid.encode()) + '.json')


def read_record(path, row):
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    if any(record.get(k) != v for k, v in row.items()):
        raise ValueError('Saved item does not match input: ' + str(path))
    if record.get('status') not in ('ok', 'error'):
        raise ValueError('Invalid saved status: ' + str(path))
    if record['status'] == 'ok' and not isinstance(record.get('answer'), str):
        raise ValueError('Invalid saved answer: ' + str(path))
    return record


def pending_rows(out, rows, retry_errors=False):
    pending = []
    for row in rows:
        path = sample_path(out, row['id'])
        old = read_record(path, row)
        if old is None or (retry_errors and old['status'] == 'error'):
            if old is not None:
                save(Path(out) / 'previous_attempts' / (path.stem + '_' + uuid.uuid4().hex + '.json'), old)
            pending.append(row)
    return pending


def fingerprint(cfg, mock):
    result = {'backend': 'mock' if mock else 'tabflash'}
    if mock:
        return result
    source, model = Path(cfg['source_dir']), Path(cfg['model_path'])
    if not (source / 'inference.py').is_file() or not (model / 'config.json').is_file():
        raise FileNotFoundError('Check source_dir and model_path in config.json')
    if 'lora' in str(model).lower() and 'merged' not in str(model).lower():
        raise ValueError('Use the merged checkpoint; this runner does not merge weights')
    files = [source / 'inference.py'] + sorted((source / 'internvl').rglob('*.py'))
    result['source_sha256'] = {str(p.relative_to(source)): digest(p.read_bytes()) for p in files}
    result['model_config_sha256'] = digest((model / 'config.json').read_bytes())
    # Record size + mtime rather than reading many GB of weights every resume.
    result['checkpoint_files'] = {
        str(p.relative_to(model)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(model.rglob('*')) if p.is_file() and '.cache' not in p.parts
    }
    result['packages'] = {}
    for name in ('torch', 'torchvision', 'transformers', 'flash-attn', 'accelerate', 'timm', 'peft'):
        try:
            result['packages'][name] = version(name)
        except PackageNotFoundError:
            result['packages'][name] = None
    return result


def worker(job):
    index, gpu, rows, cfg, out, mock = job
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if not mock:
        sys.path.insert(0, cfg['source_dir'])
        import torch
        from tqdm import tqdm
        from transformers import AutoTokenizer
        from inference import split_model, load_image
        from internvl.model.internvl_chat import InternVLChatModel, InternVLChatConfig
        torch.manual_seed(cfg['seed'])
        path = cfg['model_path']
        print(f'GPU {gpu}: loading TabFlash', flush=True)
        config = InternVLChatConfig.from_pretrained(path, local_files_only=True)
        model = InternVLChatModel.from_pretrained(
            path, config=config, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True,
            local_files_only=True, device_map=split_model(cfg['model_name'])).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False, local_files_only=True)
        iterator = tqdm(rows, desc=f'GPU {gpu}', position=index, unit='sample', dynamic_ncols=True)
    else:
        iterator = rows
    for row in iterator:
        started = time.monotonic()
        record = dict(row, status='error', gpu=str(gpu), backend='mock' if mock else 'tabflash')
        try:
            if digest(Path(row['image']).read_bytes()) != row['image_sha256']:
                raise ValueError('Image changed after preflight')
            if mock:
                response = 'MOCK: ' + row['question']
                record.update(vit_time=0.0, llm_time=0.0)
            else:
                pixels, aspect_ratio, _ = load_image(row['image'], max_num=cfg['max_dynamic_patch'])
                pixels = pixels.to(torch.bfloat16).cuda()
                with torch.inference_mode():
                    response, vit_time, llm_time, _ = model.chat(
                        tokenizer, pixels, '<image>\n' + row['question'],
                        dict(max_new_tokens=cfg['max_tokens'], do_sample=False),
                        drop_rate=cfg['drop_rate'], drop_method=cfg['drop_method'],
                        q_cond_off=cfg['q_cond_off'], prune_norm_layer_idx=cfg['prune_norm_layer_idx'])
                record.update(vit_time=float(vit_time), llm_time=float(llm_time),
                              target_aspect_ratio=list(aspect_ratio), image_patches=int(pixels.shape[0]))
                del pixels
            record['answer'] = response
            # Official chat exposes decoded text, not generated IDs/finish reason.
            record.update(requested_max_tokens=cfg['max_tokens'], finish_reason='unknown')
            if not isinstance(response, str) or not response.strip():
                raise ValueError('Empty/non-text answer preserved as error')
            record['status'] = 'ok'
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)
            if not mock:
                torch.cuda.empty_cache()
        record['seconds'] = time.monotonic() - started
        # Never catch a failed disk write as an inference error or move onward.
        save(sample_path(out, row['id']), record)
    return len(rows)


def export_results(out, rows, attempted=0):
    records = [read_record(sample_path(out, r['id']), r) for r in rows]
    saved = [r for r in records if r is not None]
    lines = [json.dumps({'question_id': r['id'], 'text': r.get('answer', '') if r['status'] == 'ok' else ''},
                        ensure_ascii=False) for r in saved]
    atomic_text(out / 'predictions.jsonl', ''.join(s + '\n' for s in lines))
    summary = {'total': len(rows), 'saved': len(saved), 'missing': len(rows) - len(saved),
               'ok': sum(r['status'] == 'ok' for r in saved),
               'errors': sum(r['status'] == 'error' for r in saved),
               'newly_attempted': attempted, 'truncated': None,
               'note': 'Official chat does not expose a reliable truncation flag. Missing items are not predictions.'}
    save(out / 'summary.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['check', 'smoke', 'full', 'export'])
    ap.add_argument('--config', type=Path, default=ROOT / 'config.json')
    ap.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    ap.add_argument('--run-name')
    ap.add_argument('--gpus', help='Comma-separated GPU IDs; may change on resume')
    ap.add_argument('--retry-errors', action='store_true')
    ap.add_argument('--per-group', type=int, default=4)
    ap.add_argument('--mock', action='store_true', help='CPU plumbing tests only, never real predictions')
    args = ap.parse_args()
    if args.per_group < 1:
        ap.error('per-group must be positive')
    name = args.run_name or (('mock-' if args.mock else '') + args.mode)
    if Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name:
        ap.error('Invalid run-name')
    if args.mock and not name.startswith('mock-'):
        ap.error('Mock runs require a mock- run-name')
    out = args.output_root / name
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        if args.mode == 'export':
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            print(json.dumps(export_results(out, manifest['rows']), indent=2))
            return
        cfg = json.loads(args.config.read_text(encoding='utf-8'))
        if args.gpus:
            cfg['gpu_ids'] = args.gpus.split(',')
        if not cfg['gpu_ids'] or len(set(cfg['gpu_ids'])) != len(cfg['gpu_ids']):
            raise ValueError('GPU list must be nonempty and unique')
        rows, ann_hash = prepare(cfg, args.mode, args.per_group)
        if not rows:
            raise ValueError('No input samples')
        identity = fingerprint(cfg, args.mock)
        print(f'Preflight OK: {len(rows)} samples', flush=True)
        if args.mode == 'check':
            return
        stable_cfg = {k: v for k, v in cfg.items() if k != 'gpu_ids'}
        manifest = {'config': stable_cfg, 'rows': rows, 'mode': args.mode,
                    'annotation_sha256': ann_hash, 'identity': identity,
                    'script_sha256': digest(Path(__file__).read_bytes()),
                    'data_script_sha256': digest((ROOT / 'data.py').read_bytes())}
        mf = out / 'manifest.json'
        if mf.exists() and json.loads(mf.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Inputs/code/environment changed; use a new --run-name')
        save(mf, manifest)
        pending = pending_rows(out, rows, args.retry_errors)
        print(f'Saved: {len(rows)-len(pending)}; pending: {len(pending)}', flush=True)
        attempted = 0
        try:
            if pending:
                with ProcessPoolExecutor(max_workers=len(cfg['gpu_ids']),
                                         mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures = [pool.submit(worker, (i, gpu, pending[i::len(cfg['gpu_ids'])], cfg, str(out), args.mock))
                               for i, gpu in enumerate(cfg['gpu_ids']) if pending[i::len(cfg['gpu_ids'])]]
                    for f in as_completed(futures):
                        attempted += f.result()
        finally:
            print(json.dumps(export_results(out, rows, attempted), indent=2), flush=True)


if __name__ == '__main__':
    main()
