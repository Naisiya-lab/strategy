"""Table-GLS official prompts, durable per-stage checkpoints, offline vLLM."""
import argparse, ast, hashlib, json, os, time
from pathlib import Path
from baseline_io import prepare, digest
ROOT = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    if os.name == "posix":
        fd = os.open(p.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

def templates():
    result = {}
    for stage in (1, 2, 3):
        path = ROOT / "official" / f"eval_mmtab_vllm_tablegls_stage{stage}.py"
        result[stage] = {n.targets[0].id: ast.literal_eval(n.value)
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id.startswith("PROMPT")}
    return result

def prompt(stage, row, prior, ts):
    ds = row["dataset_name"]
    question_sets = ("AIT-QA", "TabMCQ", "HiTab", "WTQ", "TAT-QA")
    if stage < 3:
        kw = {}
        if stage == 2:
            plan = prior.replace("```json", "").replace("```", "").strip()
            kw["reasoning_plan"] = plan if plan.endswith("}") else "N/A"
        if ds in question_sets:
            text = ts[stage]["PROMPT1"].format(question=row["original_query"], **kw)
        else:
            text = ts[stage]["PROMPT2"].format(task=row["question"].replace("e.g.,", "for example,"), **kw)
    else:
        sub = prior.strip().split("Sub-table:")[-1].strip()
        if ds in ("AIT-QA", "TabMCQ"):
            text = ts[stage]["PROMPT1"].format(subtable=sub, question=row["question"] if ds == "TabMCQ" else row["original_query"])
        elif ds in ("HiTab", "WTQ", "TAT-QA"):
            text = ts[stage]["PROMPT2"].format(subtable=sub, question=row["original_query"])
        else:
            text = ts[stage]["PROMPT3"].format(subtable=sub, task=row["question"])
    return "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + text + "<|im_end|>\n<|im_start|>assistant\n"

def process(row, cfg, out, generate, ts):
    key = digest(row["id"].encode())
    prior = ""
    stages = []
    for stage in (1, 2, 3):
        path = out / "stages" / str(stage) / (key + ".json")
        if path.exists():
            record = read(path)
            if record["id"] != row["id"] or record["stage"] != stage:
                raise ValueError("Checkpoint ID mismatch")
        else:
            text = prompt(stage, row, prior, ts)
            answer, finish, usage = generate(text, row, stage)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Empty stage {stage} response")
            record = {"id":row["id"], "stage":stage, "answer":answer,
                      "finish_reason":finish, "usage":usage, "prompt":text}
            save(path, record)
        prior = record["answer"]
        stages.append(record)
    # Extract only an unambiguous explicit answer field; retain the original stage output.
    import normalization
    fields = [o["answer"] for o in normalization.objects(prior) if "answer" in o]
    final = prior
    parsed = bool(fields) and all(v == fields[0] for v in fields)
    if parsed:
        value = fields[0]
        final = value if isinstance(value, str) else json.dumps({"answer":value}, ensure_ascii=False)
    result = {"id":row["id"], "status":"ok", "answer":final,
              "raw_answer":prior, "answer_field_extracted":parsed,
              "finish_reason":stages[-1]["finish_reason"],
              "stage_finish_reasons":[s["finish_reason"] for s in stages],
              "usage":{k:sum(s["usage"].get(k,0) for s in stages)
                       for k in ("prompt_tokens","completion_tokens","total_tokens")}}
    save(out / "samples" / (key + ".json"), result)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check","smoke","full"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--per-group", type=int, default=1)
    ap.add_argument("--run-name")
    ap.add_argument("--retry-errors", action="store_true")
    a = ap.parse_args()
    if a.per_group < 1: ap.error("per-group must be positive")
    cfg = read(a.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
    rows, ann_sha = prepare(cfg, a.mode, a.per_group)
    gold = {r["item_id"]:r for r in read(Path(cfg["data_root"])/cfg["annotation"])}
    ts = templates()
    for row in rows:
        row["original_query"] = gold[row["id"]].get("original_query")
        if row["dataset_name"] in ("AIT-QA","TabMCQ","HiTab","WTQ","TAT-QA") and not row["original_query"]:
            raise ValueError("Missing original_query: " + row["id"])
    print("Preflight:",len(rows),"samples",flush=True)
    if a.mode == "check": return
    name = a.run_name or a.mode
    if Path(name).name != name or name in (".",".."):
        raise ValueError("run-name must be a directory name")
    out = a.config.resolve().parent / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    # Kernel lock releases automatically on crash; lock file is never deleted.
    import fcntl
    with (out/".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = [Path(__file__), ROOT/"baseline_io.py", ROOT/"normalization.py", *sorted((ROOT/"official").glob("*.py"))]
        manifest = {"config":cfg,"annotation_sha256":ann_sha,"rows":rows,"mode":a.mode,
                    "source_sha256":{p.name:digest(p.read_bytes().replace(b"\r\n",b"\n")) for p in sources}}
        mf = out/"manifest.json"
        if mf.exists() and read(mf) != manifest:
            raise ValueError("Inputs/config/code changed: use a new --run-name")
        save(mf,manifest)
        pending=[]
        for row in rows:
            p=out/"samples"/(digest(row["id"].encode())+".json")
            if not p.exists() or (a.retry_errors and read(p)["status"] != "ok"):
                if p.exists(): save(out/"previous_attempts"/(p.stem+"_"+str(time.time_ns())+".json"),read(p))
                pending.append(row)
        if pending:
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
            for i,row in enumerate(pending,1):
                try: process(row,cfg,out,generate,ts)
                except Exception as e:
                    save(out/"samples"/(digest(row["id"].encode())+".json"),
                         {"id":row["id"],"status":"error","answer":"","error":repr(e)})
                    # Stop on first failure: avoid turning engine failures into thousands of empty answers.
                    raise
                print(f"Saved {i}/{len(pending)}: {row['id']}",flush=True)
        records=[read(out/"samples"/(digest(r["id"].encode())+".json")) for r in rows]
        save(out/"summary.json",{"total":len(records),"ok":sum(r["status"]=="ok" for r in records),
            "errors":sum(r["status"]!="ok" for r in records),"newly_attempted":len(pending)})
        tmp=out/"predictions.jsonl.tmp"
        with tmp.open("w",encoding="utf-8") as f:
            for r in records: f.write(json.dumps({"question_id":r["id"],"text":r["answer"],"status":r["status"]},ensure_ascii=False)+"\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,out/"predictions.jsonl")
        print("Completed:",out,flush=True)

if __name__ == "__main__": main()
