# TabFlash MMTab baseline

Run directory on machine 2: `/nc1test1/lwj/strategy/main-test/tabflash`.
Existing `code/` and `weights/` stay at their current locations. This package supplies
the runner and the existing normalized MMTab scoring implementation.

## Execution

Check `config.json` data paths first. The defaults reuse the other baselines' dataset
at `/nc1test1/lwj/data/MMTab_eval`, with `images/` and the original 49K JSON annotation.
The smoke selector uses the same seed and four samples per task/source group.

```bash
# CPU only: validates data and checkpoint/source presence; hashes images.
bash test.sh check

# Only when GPU 0 is free: one sample per group, one GPU.
bash test.sh smoke --gpus 0 --per-group 1 --run-name smoke-small

# Standard smoke, four samples per group, four single-GPU replicas.
bash test.sh smoke

# Full run, after smoke succeeds.
bash test.sh full

# Resume: repeat the same command. Saved successes AND errors are skipped.
bash test.sh full

# Explicitly retry saved errors, keeping copies in previous_attempts/.
bash test.sh full --retry-errors

# CPU only: rebuild aggregate files from existing records after a hard kill.
bash test.sh export --run-name full
```

Each completed item is written to `outputs/<run>/samples/<sha256(item_id)>.json`.
Writes flush and fsync a temporary file before atomic replacement. A stopped process
can lose its in-flight item, which is recomputed on resume; completed items are retained.
An OS lock prevents simultaneous writers and is released on process exit, including
hard termination. The `.lock` file stays on disk and must not be manually deleted.
GPU IDs may change on resume; question lists, generation settings, source, checkpoint
metadata, runner and recorded package versions must match. For changed settings, use
a new run-name. Checkpoint identity uses config hash plus file size/mtime (not full
weight hashes). Corrupt final records stop the run for inspection; temporary files
are ignored. Workers save per-item exceptions, while model-load/disk failures stop work.

`predictions.jsonl` and `summary.json` are regenerated, never appended. On hard kill,
the per-item files remain authoritative; `export` rebuilds aggregates without GPUs.
Scoring requires every selected item to have a record. Failed items count as empty
answers. No automatic answer rewriting or regeneration is performed during inference.

The model and image preprocessing are imported directly from the downloaded TabFlash
`inference.py` and modified InternVL implementation. Defaults retain `bottomk`, 0.3
drop rate, question conditioning on, 12 dynamic patches (plus the official thumbnail),
greedy decoding and a 1024-token output budget. Official chat returns text and timing
but no reliable termination reason: truncation is reported as unknown, not zero.
No weights are merged or modified. The environment is not installed/changed by test.sh.

## Scoring

Uses the normalized scorer copied from `main_test/syntab-llava-7b`, including the frozen
official metric sources and established 30-column reporting order. It needs the same
MMTab table annotations as other baselines. Reuse the existing scoring environment:

```bash
bash score.sh --run outputs/smoke
bash score.sh --run outputs/full
```

`SCORE_PYTHON` overrides the scoring interpreter. Default is
`$HOME/.conda/envs/mmtab-score/bin/python`, matching the existing baselines.
Normalized results go to `metrics_normalized/`; `--normalization raw` selects original
scoring. Mock output is explicitly refused by the scorer.

## Local validation

```bash
python -m unittest -v test_resume.py
```

CPU fixtures exercise interrupted runs, explicit error retries, preservation of completed
records, configuration mismatch rejection, corrupt-record handling, and process locks.
These tests do not establish real model loading, GPU kernel compatibility, inference
quality, or timing. Those require the remote GPU smoke run.
