"""Spawn one isolated inference process per GPU group; parent owns progress and lock."""
import multiprocessing as mp
import os, queue, signal, time
from pathlib import Path

def shard_rows(rows, count):
    return [rows[i::count] for i in range(count)]

def worker(index, gpu, cfg, rows, out, log, events):
    if os.name == "posix": os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from runtime_ui import model_output
    from evaluate import process, templates, save, digest
    with model_output(Path(log)):
        from vllm import LLM, SamplingParams
        from PIL import Image
        llm=LLM(model=cfg["model_path"],tensor_parallel_size=cfg["tensor_parallel_size"],
                gpu_memory_utilization=cfg["gpu_memory_utilization"],max_model_len=cfg["max_model_len"],
                max_num_seqs=1,seed=cfg["seed"])
        def generate(text,row,stage):
            p=Path(row["image"])
            if digest(p.read_bytes()) != row["image_sha256"]: raise ValueError("Image changed")
            with Image.open(p) as source:
                image=source.convert("RGB")
            try:
                o=llm.generate([{"prompt":text,"multi_modal_data":{"image":image}}],
                    SamplingParams(temperature=cfg["temperature"],seed=cfg["seed"],
                        max_tokens=cfg["max_tokens"][stage-1]),use_tqdm=False)[0]
            finally: image.close()
            v=o.outputs[0]; n=len(o.prompt_token_ids or []); m=len(v.token_ids)
            return v.text,v.finish_reason,{"prompt_tokens":n,"completion_tokens":m,"total_tokens":n+m}
        events.put((index, "ready", None))
        ts = templates()
        for row in rows:
            try:
                process(row, cfg, Path(out), generate, ts)
            except Exception as exc:
                save(Path(out)/"samples"/(digest(row["id"].encode())+".json"),
                     {"id":row["id"],"status":"error","answer":"","error":repr(exc)})
                raise
            events.put((index, "sample", row["id"]))
        events.put((index, "done", None))

def run_shards(cfg, rows, pending, out, logs, name, worker_fn=worker):
    from tqdm import tqdm
    groups = cfg.get("gpu_groups", [cfg["gpus"]])
    if not groups or len(set(groups)) != len(groups): raise ValueError("Invalid GPU groups")
    devices=[v.strip() for g in groups for v in g.split(',')]
    if len(devices)!=len(set(devices)): raise ValueError("GPU groups overlap")
    if any(len(g.split(',')) != cfg['tensor_parallel_size'] for g in groups):
        raise ValueError("Each GPU group must match tensor_parallel_size")
    selected={r['id'] for r in pending}
    partitions=shard_rows(rows,len(groups))
    context=mp.get_context("spawn");events=context.Queue()
    processes={}; bars={}; done=set(); paths={}
    try:
        for i, part in enumerate(partitions):
            todo=[r for r in part if r['id'] in selected]
            bars[i]=tqdm(total=len(part),initial=len(part)-len(todo),desc=f"Shard {i} | GPU {groups[i]}",
                         position=i,unit="sample",dynamic_ncols=True,mininterval=1,
                         bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            if not todo: done.add(i);continue
            paths[i]=str(Path(logs)/(f"{name}_shard{i}_model_{time.time_ns()}.log"))
            proc=context.Process(target=worker_fn,args=(i,groups[i],cfg,todo,str(out),paths[i],events))
            proc.start();processes[i]=proc
        while len(done)<len(groups):
            try:
                i,kind,value=events.get(timeout=0.2)
                if kind=="ready":
                    tqdm.write(f"Shard {i}: Model startup successful.")
                    bars[i].start_t = bars[i].last_print_t = bars[i]._time()
                    bars[i].refresh()
                elif kind=="sample":bars[i].update(1)
                elif kind=="done":done.add(i)
            except queue.Empty:
                for i,proc in processes.items():
                    if proc.exitcode is not None and i not in done:
                        raise RuntimeError(f"Shard {i} exited ({proc.exitcode}). See log: {paths[i]}")
        for i,proc in processes.items():
            proc.join(timeout=30)
            if proc.exitcode not in (0,None):raise RuntimeError(f"Shard {i} failed. See log: {paths[i]}")
    finally:
        for proc in processes.values():
            if proc.is_alive():
                if os.name == "posix":
                    try: os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError: proc.terminate()
                else:proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():proc.kill();proc.join()
        for bar in bars.values():bar.close()
        events.close()
