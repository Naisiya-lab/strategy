# ReFocus Phi-3.5-vision baseline

Evaluates the released fine-tuned checkpoint directly on original MMTab images and instructions. This is not the GPT ReFocus visual-editing agent pipeline. No strategy library is added.

## Prepare (CPU only)
Activate refocus, then run `python prepare_code.py` with network access. This downloads official Phi custom Python code into the Hugging Face cache and pins its commit in config.json. No base-model weights are downloaded. Run preparation and inference under the same user/cache. No packages are installed by this step.

## Inference
```
bash test.sh smoke --gpus 0
bash test.sh full --gpus 0,1,2,3
```
Each GPU hosts one model replica. Only select free GPUs. Use the same command to resume. GPU allocation can change on resume. Input/config/code/environment changes require a new `--run-name`.

Each finished item is saved to `outputs/<run>/samples/<id hash>.json` using flush, fsync and atomic replacement. Completed records are skipped. A crash during generation only loses the unfinished item. An OS file lock prevents concurrent writers and releases on process termination. Error records are preserved by default; explicitly retry with `--retry-errors`, which archives prior records. Corrupt records stop the run rather than being overwritten. Disk errors stop the worker.

```
bash test.sh full --gpus 0 --retry-errors
bash test.sh export --run-name full
```
Export works without loading a model, including partially completed runs. The samples directory is authoritative; predictions.jsonl is rebuilt without duplicate IDs. Scoring refuses incomplete runs and mock outputs. Do not delete the .lock file while running.

## Scoring (CPU)
Reuse the existing mmtab-score environment:
```
bash score.sh --run outputs/smoke
bash score.sh
bash score.sh --normalization raw
```
Default: outputs/full/metrics_normalized/METRICS.md and SUMMARY.tsv. Raw scores go to metrics/. Uses the same normalizer and frozen official scoring source as TabFlash and previous baselines, including separate HTML/Markdown/LaTeX TEDS. ToTTo missing references are reported explicitly. Errors remain in the denominator. No inference environment installation is needed for scoring. Set SCORE_PYTHON if your scoring Python is in a different location.

## Validation
`python -m unittest test_resume test_report` runs CPU plumbing tests, including recovery, retry archive, corruption detection and lock release. Real GPU loading and image generation remain to be validated on the remote machine.
