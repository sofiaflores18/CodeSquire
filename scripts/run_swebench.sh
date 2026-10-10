#!/usr/bin/env bash
# Run from any directory. Paths supplied as arguments are relative to repo root.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${1:-help}
if (($#)); then shift; fi
IDS=datasets/swebench_verified_validation_25_ids.txt
CONFIG=configs/qwen.yaml
DATASET=SWE-bench/SWE-bench_Verified
SPLIT=test
WORKERS=1
EVAL_WORKERS=2
TIMEOUT=600
PYTHON=${SWEBENCH_PYTHON:-.venv-swebench/bin/python}
RUN_ID="run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUT=
usage() {
  cat <<'EOF'
Usage: scripts/run_swebench.sh MODE [OPTIONS]
Modes: infer | evaluate | all | gold
  --ids FILE          One task ID per line; blank lines and # comments allowed
  --config FILE       mini-SWE-agent model/budget override
  --dataset NAME      Hugging Face dataset used by both runners
  --split NAME        Dataset split (default: test)
  --run-id NAME       Unique label; letters, numbers, underscores, hyphens
  --output DIR        Inference directory (default: runs/RUN_ID)
  --workers N         Concurrent agents (default: 1)
  --eval-workers N    Concurrent grading tasks (default: 2)
  --timeout SECONDS   Per-task test timeout (default: 600; excludes setup)
  --python PATH       Python containing swebench (or SWEBENCH_PYTHON)
Evaluate reads OUTPUT/preds.json. Gold needs no model server.
All arguments resolve from the repository root, even when called elsewhere.
EOF
}
case "$MODE" in help|-h|--help) usage; exit 0;; infer|evaluate|all|gold) ;; *) usage; exit 2;; esac
while (($#)); do
  (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
  case "$1" in
    --ids) IDS=$2;; --config) CONFIG=$2;; --dataset) DATASET=$2;;
    --split) SPLIT=$2;; --run-id) RUN_ID=$2;; --output) OUT=$2;;
    --workers) WORKERS=$2;; --eval-workers) EVAL_WORKERS=$2;;
    --timeout) TIMEOUT=$2;; --python) PYTHON=$2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
  shift 2
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
for n in "$WORKERS" "$EVAL_WORKERS" "$TIMEOUT"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo 'Budgets must be positive integers' >&2; exit 2; }
done
[[ -f "$IDS" ]] || { echo "Missing ID file: $IDS" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON (see README setup)" >&2; exit 2; }
OUT=${OUT:-runs/$RUN_ID}
# Normalize CRLF, reject duplicates/malformed IDs, and never allow an empty filter.
NORMALIZED=$("$PYTHON" - "$IDS" <<'PY'
import re, sys
from pathlib import Path
ids = [s.strip() for s in Path(sys.argv[1]).read_text().splitlines()
       if s.strip() and not s.lstrip().startswith('#')]
if not ids or len(ids) != len(set(ids)):
    sys.exit('ID list must be nonempty and contain no duplicates')
if any(not re.fullmatch(r'[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+', s) for s in ids):
    sys.exit('Malformed task ID')
print('\n'.join(ids))
PY
)
mapfile -t TASK_IDS <<< "$NORMALIZED"
FILTER=$("$PYTHON" - "${TASK_IDS[@]}" <<'PY'
import re, sys
print('^(?:' + '|'.join(map(re.escape, sys.argv[1:])) + ')$')
PY
)
command -v docker >/dev/null
docker info >/dev/null || { echo 'Docker is inaccessible; fix access before running.' >&2; exit 1; }
if [[ "$MODE" == infer || "$MODE" == all ]]; then
  command -v mini-extra >/dev/null
  [[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
  [[ ! -e "$OUT" ]] || { echo "Output exists: $OUT. Choose a new output/run ID." >&2; exit 2; }
fi
if [[ "$MODE" != infer ]]; then
  "$PYTHON" -c 'import swebench.harness.run_evaluation'
  [[ ! -e "logs/run_evaluation/$RUN_ID" ]] || { echo 'Evaluation run ID already exists; choose a new one.' >&2; exit 2; }
fi
echo "Mode=$MODE tasks=${#TASK_IDS[@]} dataset=$DATASET split=$SPLIT run=$RUN_ID"
mkdir -p "$OUT"
printf '%s\n' "${TASK_IDS[@]}" > "$OUT/selected_ids.txt"
if [[ "$MODE" == infer || "$MODE" == all ]]; then
  cp -- "$CONFIG" "$OUT/model-config.yaml"
  # Record dataset identity and budgets without recording credentials/config contents.
  printf '%s\n' "dataset=$DATASET" "split=$SPLIT" "workers=$WORKERS" "eval_workers=$EVAL_WORKERS" "timeout=$TIMEOUT" > "$OUT/run-settings.txt"
  mini-extra --version > "$OUT/mini-version.txt" 2>&1 || true
  mini-extra swebench --subset "$DATASET" --split "$SPLIT" \
    --filter "$FILTER" --workers "$WORKERS" \
    -c swebench.yaml -c "$CONFIG" --output "$OUT" \
    2>&1 | tee "$OUT/inference-console.log"
fi
if [[ "$MODE" == infer || "$MODE" == all || "$MODE" == evaluate ]]; then
  # Preserve partial failures: count empty patches rather than silently calling them success.
  "$PYTHON" - "$OUT/preds.json" "${TASK_IDS[@]}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file(): sys.exit(f'Missing predictions: {p}')
x = json.loads(p.read_text())
rows = list(x.values()) if isinstance(x, dict) else x
got = {r['instance_id'] for r in rows}
expected = set(sys.argv[2:])
if got != expected:
    sys.exit(f'Prediction IDs differ: missing={sorted(expected-got)}, extra={sorted(got-expected)}')
count = sum(bool(r.get('model_patch', '').strip()) for r in rows)
print(f'Patches: {count}/{len(rows)} nonempty; {len(rows)-count} empty')
if count == 0: sys.exit('No nonempty patches. Inspect trajectories before grading.')
PY
fi
if [[ "$MODE" == evaluate || "$MODE" == all || "$MODE" == gold ]]; then
  PREDICTIONS="$OUT/preds.json"
  [[ "$MODE" != gold ]] || PREDICTIONS=gold
  "$PYTHON" -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET" --split "$SPLIT" \
    --predictions_path "$PREDICTIONS" --instance_ids "${TASK_IDS[@]}" \
    --max_workers "$EVAL_WORKERS" --timeout "$TIMEOUT" --run_id "$RUN_ID" \
    2>&1 | tee "$OUT/evaluation-console.log"
fi
echo "Finished commands. Inspect trajectories, exit statuses and grading reports; completion does not imply resolved tasks."
