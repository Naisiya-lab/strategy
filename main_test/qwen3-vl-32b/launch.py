"""Linux launcher: two independent tensor-parallel replicas, each on two GPUs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent


def command(cfg, index):
    from urllib.parse import urlparse
    port=urlparse(cfg['endpoints'][index]).port
    return [sys.executable,'-m','vllm.entrypoints.openai.api_server',
            '--model',cfg['model_path'],'--served-model-name',cfg['served_model'],
            '--host','127.0.0.1','--port',str(port),'--dtype','bfloat16',
            '--tensor-parallel-size',str(cfg['tensor_parallel_size']),
            '--gpu-memory-utilization',str(cfg['gpu_memory_utilization']),
            '--max-model-len',str(cfg['server_max_model_len']),
            '--max-num-seqs',str(cfg['max_num_seqs']),
            '--max-num-batched-tokens',str(cfg['max_num_batched_tokens']),
            '--limit-mm-per-prompt','{"image":1,"video":0}',
            '--generation-config','vllm']


def validate(cfg):
    if len(cfg['endpoints'])!=len(cfg['gpu_groups']) or len(cfg['internal_ports'])!=len(cfg['endpoints']):
        raise ValueError('Each endpoint must have a GPU group and internal port')
    gpus=[g.strip() for group in cfg['gpu_groups'] for g in group.split(',')]
    if len(gpus)!=len(set(gpus)): raise ValueError('GPU groups overlap')
    if any(len(group.split(','))!=cfg['tensor_parallel_size'] for group in cfg['gpu_groups']):
        raise ValueError('GPU count must equal tensor_parallel_size')
    if len(set(cfg['endpoints']))!=len(cfg['endpoints']): raise ValueError('Duplicate endpoint')
    if len(set(cfg['internal_ports']))!=len(cfg['internal_ports']): raise ValueError('Duplicate internal port')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['start','stop'])
    ap.add_argument('--replica',type=int,choices=[0,1],help='Only operate on this replica')
    ap.add_argument('--startup-timeout',type=int,default=900)
    args=ap.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf8'));validate(cfg)
    indices=list(range(len(cfg['endpoints']))) if args.replica is None else [args.replica]
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'pids').mkdir(exist_ok=True)
    if args.action=='stop':
        for i in indices:
            p=ROOT/'pids'/f'replica_{i}.json'
            if not p.exists(): continue
            rec=json.loads(p.read_text());pid=rec['pid'];proc=Path(f'/proc/{pid}/cmdline')
            if not proc.exists():
                print('Leader no longer running:',pid,'; check nvidia-smi for remaining workers');continue
            cmd=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'vllm.entrypoints.openai.api_server' not in cmd or cfg['model_path'] not in cmd or f"--port {rec['port']} " not in cmd:
                raise RuntimeError('PID no longer matches owned server; refusing to stop it')
            if os.getpgid(pid)!=pid: raise RuntimeError('Unexpected process group')
            os.killpg(pid,signal.SIGTERM)
            print('Sent shutdown to replica',i,'process group',pid,flush=True)
        return
    if args.startup_timeout<1: ap.error('startup-timeout must be positive')
    if not Path(cfg['model_path'],'config.json').is_file(): raise ValueError('Model weights not found')
    from transformers import AutoTokenizer
    from vllm.transformers_utils.tokenizer import get_cached_tokenizer
    get_cached_tokenizer(AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True))
    # Check all selected API/internal start ports before allocating any model.
    for i in indices:
        for port in (int(command(cfg,i)[command(cfg,i).index('--port')+1]),cfg['internal_ports'][i]):
            with socket.socket() as s: s.bind(('',port))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # Wait for one replica to become ready before starting the next to avoid
    # concurrent initialization races and simultaneous peak loading pressure.
    for i in indices:
        cmd=command(cfg,i);port=int(cmd[cmd.index('--port')+1])
        log=ROOT/'logs'/f'server_{port}_{datetime.now():%Y%m%d_%H%M%S}.log'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['gpu_groups'][i],
                 VLLM_PORT=str(cfg['internal_ports'][i]),VLLM_HOST_IP='127.0.0.1')
        with log.open('ab') as stream:
            proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,cwd=ROOT,env=env,start_new_session=True)
        record={'pid':proc.pid,'port':port,'gpu_group':cfg['gpu_groups'][i],
                'internal_port':cfg['internal_ports'][i],'command':cmd,'log':str(log)}
        (ROOT/'pids'/f'replica_{i}.json').write_text(json.dumps(record,indent=2))
        print('Starting replica',i,'GPUs',cfg['gpu_groups'][i],'API',port,'log',log,flush=True)
        deadline=time.monotonic()+args.startup_timeout;ticks=0
        while time.monotonic()<deadline:
            if proc.poll() is not None: raise RuntimeError(f'Replica {i} exited; see {log}')
            try:
                with opener.open(cfg['endpoints'][i]+'/models',timeout=3) as response:
                    models=json.load(response)
                if cfg['served_model'] in [r['id'] for r in models['data']]:
                    print('READY',cfg['endpoints'][i],flush=True);break
            except (OSError,ValueError,KeyError): pass
            if ticks%5==0: print('Waiting for replica',i,'to load/compile...',flush=True)
            time.sleep(2);ticks+=1
        else: raise RuntimeError(f'Replica {i} startup timed out; process left running, inspect {log} before retrying')


if __name__=='__main__':main()
