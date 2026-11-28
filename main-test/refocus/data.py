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
        # IDs such as TabFact_*.html.csv are opaque identifiers. Only strip
        # actual image suffixes, never .csv/.html components of an ID.
        image_name = Path(iid).name
        key = Path(image_name).stem if Path(image_name).suffix.lower() in {
            '.jpg', '.jpeg', '.png', '.webp', '.bmp'} else image_name
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
