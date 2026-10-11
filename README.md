# CodeSquire SWE-bench runner

Evaluation harness for the CodeSquire agent. To *use* the agent on your own code instead
of benchmarking it, see [cli/README.md](cli/README.md):

```bash
./cli/codesquire.py "add type hints to utils.py"
```

## Instance setup

Install Docker with working non-root access, uv, and the NVIDIA driver required by your
vLLM installation. This runner assumes the model server is already running; gold checks
do not use the model server.

```bash
uv tool install 'mini-swe-agent==2.4.6'
uv venv .venv-swebench --python 3.12
uv pip install --python .venv-swebench/bin/python swebench
chmod +x scripts/run_swebench.sh
docker info
curl http://localhost:8000/v1/models
```


Pin vLLM separately in the model-server setup. My local tested server used 0.31.0,
FP16, an 8192-token total context, and the Hermes tool parser. Record the exact server
command and model revision with experiments.

## Commands

One task, inference and grading:

```bash
scripts/run_swebench.sh all \
  --ids datasets/smoke_ids.txt \
  --config configs/qwen.yaml \
  --run-id qwen_smoke_01
```

Reference-patch validation for all selected tasks:

```bash
scripts/run_swebench.sh gold --run-id verified25_gold_01
```

Inference for all 25, then grading as a separate stage:

```bash
scripts/run_swebench.sh infer --run-id qwen25_01
scripts/run_swebench.sh evaluate \
  --output runs/qwen25_01 \
  --run-id qwen25_01_eval
```

Different model, selection, or dataset:

```bash
scripts/run_swebench.sh all \
  --dataset SWE-bench/SWE-bench_Verified \
  --split test \
  --ids datasets/my_selection_ids.txt \
  --config configs/another_model.yaml \
  --workers 1 --eval-workers 2 --timeout 600 \
  --run-id another_model_01
```

Use the same ID file when running inference and grading separately. The script checks
prediction IDs against that file. Nonempty predictions are still graded if some tasks
have empty patches; inspect and count those failures in your comparisons. All-empty
predictions stop the runner with an actionable error. Failed per-task agent attempts
can appear in an otherwise successful CLI process; read the saved exit-status files.

Inference output directories and evaluation run IDs must be new. To retry, choose a
new run ID; this prevents accidental reuse of cached failures or stale reports.
`--timeout` applies to test execution, not image downloads or the whole agent attempt.
The 30-step config is a step budget, not a wall-clock budget. Run IDs label experiments;
the harness writes reports at the repository root and detailed logs under `logs/`.

The manifest records a pinned upstream revision, but this runner uses the live dataset
name and split. It does not claim revision-pinned evaluation. For exact reproducibility,
prepare a full-row dataset snapshot at that revision and use a compatible dataset path
for both tools; the selection manifest itself is not a full-row dataset. Selected IDs
absent from the live dataset will be caught by the prediction-ID check after inference.
