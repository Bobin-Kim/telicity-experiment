# Reviewer-response robustness run

Check progress:

`/data/sl_datasets/wlasl/Qwen2.5-1.5B-Instruct/venv/bin/python results/reviewer_response_robustness/code/status.py`

The detached supervisor writes `logs/supervisor.log`, per-model logs under `logs/`, worker JSON under `status/`, resumable per-fold shards under each experiment family, and `FINAL_REPORT.md` after all 25 model/fold units in each family pass aggregation.
