# GLM-5.3-Flash full MMTab baseline

Direct original image + original question; no gold tables/answers, strategy memory, agents or answer repair in the request. Reuses historical GLM profile: thinking enabled, reasoning_effort high, max_tokens 8192. No new temperature/seed sampling options are sent. HTTP response uses non-streaming transport.

## Machine 2

Python >=3.10, tqdm for inference progress. No GPU or model installation needed.
Data: /nc1test1/lwj/data/MMTab_eval (same as Qwen); annotation MMTab-eval_test_data_49K.json; images/; scoring also needs MMTab-eval_test_tables_23K.json.

```bash
cd /nc1test1/lwj/strategy/main_test/glm_5.3_flash
python -m pip install -r requirements.txt
read -rsp 'ZHIPU_API_KEY: ' ZHIPU_API_KEY; echo
export ZHIPU_API_KEY
bash test.sh smoke --concurrency 20
bash score.sh --run outputs/smoke
bash test.sh full --concurrency 20
bash score.sh
```

Smoke selects up to 100 samples deterministically across task/dataset groups. Full uses every annotation ID, including OOD and ToTTo. ToTTo without references is reported unscored. Full and smoke are independent and do not reuse each other's results. Preflight scans images and verifies hashes, printing progress every 1000 rows. `bash test.sh check` is optional and makes no API calls.

Inference displays an ASCII tqdm bar with elapsed time, ETA, throughput and session success/error counts, saves raw responses (including reasoning if returned), token usage and elapsed time under outputs/<mode>/samples, then exports predictions.jsonl and summary.json. Protect these local results as research data; outputs and credentials are excluded from Git.

Default concurrency 20; pass --concurrency 30 or 40 after checking service stability. Accepted range 1–50 is a per-process limit; other experiments share the account quota. This runner does not automatically increase concurrency. HTTP 429/temporary server errors use a shared cooldown, Retry-After (seconds or HTTP date), exponential backoff/jitter otherwise, with four retries. Persistent authentication/request errors stop new submissions. There is no automatic semantic regeneration. Nonempty truncated answers are retained and flagged, empty responses are errors. Timeout retries may incur repeated billing; reported usage may be incomplete.

Rerun the same command to resume. Saved errors are skipped unless --retry-errors is supplied:

```bash
bash test.sh full --concurrency 30 --retry-errors
```

Changing concurrency is permitted during resume; model/input/code changes require --run-name NEW_NAME. Each output directory has a lock. After a forced kill, remove outputs/full/.lock only after verifying the old process is no longer running. Ctrl+C stops new submissions and waits for in-flight requests to save (up to the configured timeout). Separate output names must not be run concurrently beyond the shared API limit.

Scoring reuses the latest main_test normalization and frozen official metric code, preserving 30-column order and HTML/Markdown/LaTeX TEDS. Existing mmtab-score environment can be reused. If absent, run `bash install_score.sh`; SCORE_PYTHON can override its Python path.

```bash
bash score.sh                         # outputs/full/metrics_normalized
bash score.sh --normalization raw     # outputs/full/metrics
```

Scoring requires a saved record for every selected sample; failed records remain in the denominator. All original response files are retained. Offline tests: `python3 -m unittest -v test_local.py`.
