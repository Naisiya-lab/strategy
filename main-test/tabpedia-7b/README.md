# TabPedia MMTab-eval baseline

Remote directory: /nc1test1/lwj/strategy/main-test/tabpedia-7b
Use the existing tabpedia Conda environment and official code/weights/components.
No training or strategy-library modules are used. MMTab input text is kept unchanged;
only official image tokens, InternLM2 chat formatting and learned tokens are added.
Official dual-image preprocessing is reused. Reference answers never enter inference.

## Commands

```bash
bash test.sh smoke --gpus 0
bash test.sh smoke --gpus 0,1,2,3
bash test.sh full --gpus 0,1,2,3
# Resume: repeat the exact smoke/full command; GPU allocation may change.
bash test.sh full --retry-errors
# Export an interrupted run without loading a model:
bash test.sh export --run-name full
bash install_score.sh
bash score.sh --run outputs/smoke
bash score.sh
bash score.sh --normalization raw
```

Smoke selects four items per (task_type,dataset_name), seed 42, using the same
selection as existing baselines (368 for the current dataset). Full uses every
annotation. Check config.json data_root against your existing Qwen configuration.
Default four workers, one complete model per GPU. No model service is required.

Each completed or failed item is atomically saved and fsynced under samples/.
Reruns skip all completed attempts, including errors, unless --retry-errors is given.
Retries archive old records under previous_attempts/. Corrupt records are rejected.
An OS lock prevents concurrent writers and releases on process death. No manual
lock-file deletion is needed. Changed input/config/code/environment requires a new
--run-name. Large model files are tracked by size and modification time, not full
content hashes. The annotation, images, runner and backend are hashed.

Default generation: deterministic, maximum 3000 output tokens, repetition penalty
1.1, matching official QA settings. Budget is reduced to fit native/config context
and recorded per item. Inputs are never silently truncated. Empty answers and
inference exceptions are saved as errors. Checkpoint key mismatches stop startup
for review rather than silently accepting partial weights.

Outputs: manifest.json, samples/*.json, predictions.jsonl, summary.json.
Scoring uses the same frozen official MMTab functions and V2 normalization as
existing baselines. metrics_normalized/ is the default; raw scores use metrics/.
Both contain METRICS.md, SUMMARY.tsv (30 ordered columns), metrics.json. All three
TR formats report separate TEDS; failed conversions count as zero. Missing ToTTo
references are explicitly unavailable. Incomplete runs and mock runs cannot score.

CPU tests: python -m unittest discover -p 'test_*.py'
GPU loading and real answer quality are not yet verified. All MMTab tasks are
adapted as image-plus-instruction generation; this is not a claim that the original
TabPedia benchmark covered every MMTab task.
