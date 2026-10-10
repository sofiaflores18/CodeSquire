#!/usr/bin/env bash
# Docker-free SWE-bench inference on this machine; optional cloud grading with sb-cli.
# Paths are relative to CodeSquire's repository root.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODE=${1:-help}
if (($#)); then shift; fi

IDS=datasets/swebench_verified_validation_25_ids.txt
CONFIG=configs/qwen.yaml
DATASET=princeton-nlp/SWE-Bench_Verified
SPLIT=test
WORKERS=1
TIMEOUT=600
MODEL_ALIAS=
RUN_ID="run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUT=

usage() {
  cat <<'EOF'
Usage: scripts/run_swebench.sh MODE [OPTIONS]
Modes: infer | evaluate | all
  --model ALIAS       Optional alias from models.yaml (e.g. qwen-4b)
  --ids FILE          Exact SWE-bench instance IDs (one per line)
  --config FILE       mini-SWE-agent model/budget overrides
  --dataset NAME      HF dataset (default: princeton-nlp/SWE-Bench_Verified)
  --split NAME        Dataset split (default: test)
  --run-id NAME       Unique run label
  --output DIR        Run directory (default: runs/RUN_ID)
  --workers 1         Must remain 1: one shared local worktree
  --timeout SECONDS   Timeout per shell command (default: 600)

infer:    Generate 25 predictions using mini-SWE-agent, WITHOUT Docker.
evaluate: Submit existing OUTPUT/preds.json to SWE-bench cloud (sb-cli).
all:      Inference, then submit predictions for cloud evaluation.

vLLM must already be serving the model on OPENAI_API_BASE.
Cloud grading requires sb-cli and SWEBENCH_API_KEY.
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
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift 2
done

[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid run ID" >&2; exit 2; }
[[ "$WORKERS" == 1 ]] || { echo "--workers must be 1 in local mode" >&2; exit 2; }
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "Timeout must be positive" >&2; exit 2; }
[[ -f "$IDS" ]] || { echo "Missing ID file: $IDS" >&2; exit 2; }
command -v python >/dev/null

# Validate and normalize exact task IDs; reject duplicates and empty selections.
mapfile -t TASK_IDS < <(python - "$IDS" <<'PY'
import re, sys
from pathlib import Path
ids = [s.strip() for s in Path(sys.argv[1]).read_text().splitlines()
       if s.strip() and not s.lstrip().startswith('#')]
if not ids or len(ids) != len(set(ids)):
    sys.exit('Task IDs must be nonempty and unique')
if any(not re.fullmatch(r'[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+', s) for s in ids):
    sys.exit('Malformed task ID')
print('\n'.join(ids))
PY
)
((${#TASK_IDS[@]} > 0)) || { echo "No task IDs found" >&2; exit 2; }
FILTER=$(python - "${TASK_IDS[@]}" <<'PY'
import re, sys
print('^(?:' + '|'.join(map(re.escape, sys.argv[1:])) + ')$')
PY
)

if [[ "$MODE" == evaluate && -z "$OUT" ]]; then
  echo "Pass --output runs/EXISTING_RUN when evaluating saved predictions" >&2
  exit 2
fi
OUT=${OUT:-runs/$RUN_ID}

if [[ "$MODE" == infer || "$MODE" == all ]]; then
  command -v mini-extra >/dev/null || { echo "mini-extra not installed" >&2; exit 1; }
  command -v git >/dev/null || { echo "git not installed" >&2; exit 1; }
  [[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 2; }
  [[ -f "$CONFIG" ]] || { echo "Missing mini-SWE-agent config: $CONFIG" >&2; exit 2; }
  [[ -n "${OPENAI_API_BASE:-}" ]] || { echo "Set OPENAI_API_BASE to your vLLM endpoint" >&2; exit 2; }
  [[ ! -e /testbed && ! -L /testbed ]] || {
    echo "/testbed already exists; won't overwrite another workspace" >&2; exit 1;
  }
  # The stock SWE-bench prompt explicitly uses /testbed. Give it a dedicated
  # symlink into this run's worktree so paths match while avoiding /testbed data.
  mkdir -p "$OUT/worktree"
  OUT_ABS=$(cd "$OUT" && pwd)
  ln -s "$OUT_ABS/worktree" /testbed
  trap '[[ -L /testbed ]] && rm -- /testbed' EXIT

  # Loaded *after* the standard SWE-bench config and model overrides.
  # Each instance starts from its own clean base commit; jobs MUST be sequential.
  cat > "$OUT/local-environment.yaml" <<EOF
  environment:
    environment_class: local
    cwd: /testbed
    timeout: $TIMEOUT
  run:
    env_startup_command: |
      set -eu
      find . -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
      git clone --filter=blob:none --no-checkout "https://github.com/{{ repo }}.git" .
      git checkout --detach "{{ base_commit }}"
EOF
  printf '%s\n' "${TASK_IDS[@]}" > "$OUT/selected_ids.txt"
  cp -- "$CONFIG" "$OUT/model-config.yaml"
  printf '%s\n' "dataset=$DATASET" "split=$SPLIT" "workers=1" "timeout=$TIMEOUT" > "$OUT/run-settings.txt"

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
  echo "Local inference: ${#TASK_IDS[@]} tasks, run=$RUN_ID, output=$OUT"
  mini-extra swebench --subset "$DATASET" --split "$SPLIT" \
    --filter "$FILTER" --workers 1 --environment-class local \
    -c swebench.yaml -c "$CONFIG" -c "$OUT/local-environment.yaml" \
    "${MODEL_ARGS[@]}" --output "$OUT" \
    2>&1 | tee "$OUT/inference-console.log"
fi

# Verify that the runner returned predictions for every selected ID.
python - "$OUT/preds.json" "${TASK_IDS[@]}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file(): sys.exit(f'Missing predictions: {p}')
x = json.loads(p.read_text())
rows = list(x.values()) if isinstance(x, dict) else x
got = {r['instance_id'] for r in rows}
expected = set(sys.argv[2:])
if len(rows) != len(expected) or got != expected:
    sys.exit(f'Prediction mismatch: missing={sorted(expected-got)}, extra={sorted(got-expected)}')
count = sum(bool((r.get('model_patch') or '').strip()) for r in rows)
print(f'Nonempty patches: {count}/{len(rows)}')
if count == 0: sys.exit('No nonempty patches; inspect the trajectories')
PY

if [[ "$MODE" == evaluate || "$MODE" == all ]]; then
  command -v sb-cli >/dev/null || { echo "Install cloud grader: uv pip install sb-cli" >&2; exit 1; }
  [[ -n "${SWEBENCH_API_KEY:-}" ]] || { echo "Set SWEBENCH_API_KEY before cloud grading" >&2; exit 1; }
  mkdir -p "$OUT/cloud-report"
  IDS_CSV=$(IFS=,; echo "${TASK_IDS[*]}")
  sb-cli submit swe-bench_verified "$SPLIT" \
    --predictions_path "$OUT/preds.json" \
    --instance_ids "$IDS_CSV" \
    --run_id "$RUN_ID" \
    --output_dir "$OUT/cloud-report" \
    2>&1 | tee "$OUT/evaluation-console.log"
fi

echo "Finished. Predictions: $OUT/preds.json; trajectories: $OUT/<instance_id>/"
