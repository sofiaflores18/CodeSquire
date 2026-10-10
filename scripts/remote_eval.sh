#!/usr/bin/env bash
# SWE-bench inference and grading with Modal sandboxes; no local Docker or sb-cli.
# Run from any directory. File paths are relative to the CodeSquire repo root.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODE=${1:-help}
if (($#)); then shift; fi

IDS=datasets/swebench_verified_validation_25_ids.txt
CONFIG=configs/qwen.yaml
DATASET=princeton-nlp/SWE-bench_Verified
SPLIT=test
WORKERS=1
TIMEOUT=600
MODEL_ALIAS=
PYTHON=${SWEBENCH_PYTHON:-.venv-swebench/bin/python}
RUN_ID="run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUT=

usage() {
  cat <<'EOF'
Usage: scripts/remote_eval.sh MODE [OPTIONS]
Modes: infer | evaluate | all
  --model ALIAS       Optional model alias from models.yaml (e.g. qwen-4b)
  --ids FILE          SWE-bench task IDs, one per line
  --config FILE       mini-SWE-agent model/budget overrides
  --dataset NAME      HF dataset (default: princeton-nlp/SWE-bench_Verified)
  --split NAME        Dataset split (default: test)
  --run-id NAME       Unique evaluation run ID
  --output DIR        Run directory (default: runs/RUN_ID)
  --workers N         Concurrent mini-SWE-agent workers (default: 1)
  --timeout SECONDS   Per-agent shell command and per-task grading timeout (default: 600)
  --python PATH       Python with swebench[modal] (default: .venv-swebench/bin/python)

infer:    mini-SWE-agent + Modal sandboxes -> OUTPUT/preds.json
 evaluate: Official SWE-bench grading on Modal of existing OUTPUT/preds.json
all:      Both stages; collect reports into OUTPUT/evaluation/

Before running: configure Modal with `modal setup` and have vLLM serving
at OPENAI_API_BASE for infer/all. Install mini-swe-agent[modal] and swebench[modal].
No Docker daemon or SWEBENCH_API_KEY is needed.
EOF
}

case "$MODE" in
  infer|evaluate|all) ;;
  help|-h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

while (($#)); do
  (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
  case "$1" in
    --model) MODEL_ALIAS=$2 ;;
    --ids) IDS=$2 ;;
    --config) CONFIG=$2 ;;
    --dataset) DATASET=$2 ;;
    --split) SPLIT=$2 ;;
    --run-id) RUN_ID=$2 ;;
    --output) OUT=$2 ;;
    --workers) WORKERS=$2 ;;
    --timeout) TIMEOUT=$2 ;;
    --python) PYTHON=$2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift 2
done

[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid run ID" >&2; exit 2; }
for n in "$WORKERS" "$TIMEOUT"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "Worker count/timeout must be positive integers" >&2; exit 2; }
done
[[ -f "$IDS" ]] || { echo "Missing ID file: $IDS" >&2; exit 2; }

# Command substitution propagates malformed-ID/duplicate checks (unlike process substitution).
NORMALIZED=$(python - "$IDS" <<'PY'
import re, sys
from pathlib import Path
ids = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines()
       if line.strip() and not line.lstrip().startswith('#')]
if not ids or len(ids) != len(set(ids)):
    sys.exit('Task IDs must be nonempty and unique')
if any(not re.fullmatch(r'[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+', value) for value in ids):
    sys.exit('Malformed task ID')
print('\n'.join(ids))
PY
)
mapfile -t TASK_IDS <<< "$NORMALIZED"
FILTER=$(python - "${TASK_IDS[@]}" <<'PY'
import re, sys
print('^(?:' + '|'.join(map(re.escape, sys.argv[1:])) + ')$')
PY
)

if [[ "$MODE" == evaluate && -z "$OUT" ]]; then
  echo "Pass --output runs/EXISTING_RUN for saved predictions" >&2
  exit 2
fi
OUT=${OUT:-runs/$RUN_ID}
if [[ "$MODE" == infer || "$MODE" == all ]]; then
  [[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 2; }
  [[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
  command -v mini-extra >/dev/null || { echo "Install mini-swe-agent[modal]" >&2; exit 1; }
  [[ -n "${OPENAI_API_BASE:-}" ]] || { echo "Set OPENAI_API_BASE to vLLM's OpenAI-compatible endpoint" >&2; exit 2; }
else
  [[ -f "$OUT/preds.json" ]] || { echo "Missing predictions: $OUT/preds.json" >&2; exit 2; }
  if [[ -f "$OUT/selected_ids.txt" ]]; then
    diff -u "$OUT/selected_ids.txt" <(printf '%s\n' "${TASK_IDS[@]}") || {
      echo "Requested task IDs differ from the inference run; pass the matching --ids file" >&2
      exit 2
    }
  fi
fi

# Modal authentication is used by both mini-SWE-agent and the SWE-bench grader.
if [[ "$MODE" == all || "$MODE" == evaluate ]]; then
  [[ -x "$PYTHON" ]] || { echo "Missing grader Python: $PYTHON" >&2; exit 1; }
  "$PYTHON" -c 'import swebench.harness.run_evaluation, modal' || {
    echo "Install swebench[modal] in $PYTHON" >&2; exit 1;
  }
fi
if [[ ! -f "${HOME}/.modal.toml" ]]; then
  echo "Modal login not found (~/.modal.toml). Run: modal setup" >&2
  exit 1
fi

if [[ "$MODE" == infer || "$MODE" == all ]]; then
  mkdir -p "$OUT"
  printf '%s\n' "${TASK_IDS[@]}" > "$OUT/selected_ids.txt"
  cp -- "$CONFIG" "$OUT/model-config.yaml"
  printf '%s\n' "dataset=$DATASET" "split=$SPLIT" "workers=$WORKERS" \
    "timeout=$TIMEOUT" "agent_backend=swerex_modal" "grader_backend=modal" > "$OUT/run-settings.txt"

  # SWE-bench images already contain /testbed at the appropriate base commit.
  # Do not clone repos manually, create a /testbed symlink, or run startup cleanup.
  # This override is applied LAST so a prior local-mode config can't force Docker.
  cat > "$OUT/modal-environment.yaml" <<EOF
environment:
  environment_class: swerex_modal
  cwd: /testbed
  timeout: $TIMEOUT
  startup_timeout: 600.0
  runtime_timeout: 10800.0
  deployment_timeout: 10800.0
run:
  env_startup_command: ""
EOF

  MODEL_ARGS=()
  if [[ -n "$MODEL_ALIAS" ]]; then
    [[ -f models.yaml ]] || { echo "Missing models.yaml" >&2; exit 2; }
    HF_ID=$(python - "$MODEL_ALIAS" <<'PY'
import sys, yaml
models = yaml.safe_load(open('models.yaml'))['models']
if sys.argv[1] not in models:
    sys.exit(f'Unknown model alias: {sys.argv[1]}')
print(models[sys.argv[1]])
PY
)
    MODEL_ARGS=(-m "openai/$HF_ID")
  fi

  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
  echo "Modal agent inference: ${#TASK_IDS[@]} instances; run=$RUN_ID"
  mini-extra swebench --subset "$DATASET" --split "$SPLIT" \
    --filter "$FILTER" --workers "$WORKERS" --environment-class swerex_modal \
    -c swebench.yaml -c "$CONFIG" -c "$OUT/modal-environment.yaml" \
    "${MODEL_ARGS[@]}" --output "$OUT" \
    2>&1 | tee "$OUT/inference-console.log"
fi

# Verify complete prediction coverage, including empty patches as real attempts.
python - "$OUT/preds.json" "${TASK_IDS[@]}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file(): sys.exit(f'Missing predictions: {path}')
obj = json.loads(path.read_text())
rows = list(obj.values()) if isinstance(obj, dict) else obj
if not isinstance(rows, list) or not all(isinstance(x, dict) for x in rows):
    sys.exit('Invalid predictions: expected dictionary/list of prediction objects')
got = [r.get('instance_id') for r in rows]
expected = set(sys.argv[2:])
if len(got) != len(expected) or set(got) != expected:
    sys.exit(f'Prediction mismatch: missing={sorted(expected - set(got))}, extra={sorted(set(got) - expected)}')
count = sum(bool((r.get('model_patch') or '').strip()) for r in rows)
print(f'Nonempty patches: {count}/{len(rows)}')
if count == 0: sys.exit('No nonempty patches; inspect the trajectories')
PY

if [[ "$MODE" == evaluate || "$MODE" == all ]]; then
  # The official harness creates logs/run_evaluation/RUN_ID/results.json.
  # Use a unique run ID for every distinct predictions file to avoid cached grades.
  LOG_DIR="logs/run_evaluation/$RUN_ID"
  [[ ! -e "$LOG_DIR" ]] || {
    echo "Evaluation run ID already has logs: $LOG_DIR. Choose a new --run-id" >&2
    exit 2
  }
  mkdir -p "$OUT/evaluation"
  echo "Official SWE-bench grading on Modal: $RUN_ID"
  "$PYTHON" -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET" --split "$SPLIT" \
    --predictions_path "$OUT/preds.json" \
    --instance_ids "${TASK_IDS[@]}" \
    --timeout "$TIMEOUT" --run_id "$RUN_ID" --modal true \
    2>&1 | tee "$OUT/evaluation-console.log"

  [[ -f "$LOG_DIR/results.json" ]] || {
    echo "Grading returned without a results.json; inspect evaluation-console.log" >&2
    exit 1
  }
  cp -- "$LOG_DIR/results.json" "$OUT/evaluation/results.json"
  # Keep small per-instance grading reports in the pushable output folder.
  while IFS= read -r -d '' report; do
    rel=${report#"$LOG_DIR"/}
    mkdir -p "$OUT/evaluation/$(dirname "$rel")"
    cp -- "$report" "$OUT/evaluation/$rel"
  done < <(find "$LOG_DIR" -type f -name report.json -print0)
  python - "$OUT/evaluation/results.json" "${#TASK_IDS[@]}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
if r.get('total_instances') != expected:
    sys.exit(f"Grading selected {r.get('total_instances')} instances; expected {expected}")
resolved = r.get('resolved_instances', 0)
completed = r.get('completed_instances', 0)
errors = r.get('error_instances', 0)
empty = r.get('empty_patch_instances', 0)
print(f"SWE-bench: {resolved}/{expected} resolved; "
      f"{completed} completed, {errors} grading errors, {empty} empty patches")
if completed + errors + empty < expected:
    sys.exit('Incomplete evaluation: some tasks lack results. Inspect Modal logs.')
PY
fi

echo "Finished. Predictions: $OUT/preds.json; grading: $OUT/evaluation/results.json (if evaluated)"
