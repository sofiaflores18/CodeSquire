# Eval: 25 SWE-bench Verified tasks, one model

```
python run_eval.py --model qwen3-8b
        │
        ├─ reads   models.yaml                 alias → HF model id
        ├─ reads   configs/<alias>.yaml        context, steps, sampling, thinking
        ├─ reads   datasets/swebench_verified_validation_25.json   the 25 task ids
        │
        ├─ 1. starts vLLM on this GPU (port 8000)
        ├─ 2. scripts/swebench_runner.py       agent runs each task in a Modal sandbox
        │                                      (+ stops sandboxes, ends repeat-command loops)
        ├─ 3. stops vLLM, kills leftover sandboxes
        └─ 4. scripts/grade_run.py             grades nonempty patches on Modal
                                               → runs/<alias>_<timestamp>/report.json
```

## Run

```bash
cd /workspace/CodeSquire && source /venv/main/bin/activate
python run_eval.py --model qwen3-8b
```

One eval at a time: each vLLM server takes most of the GPU and port 8000.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | Alias from `models.yaml`; needs a matching `configs/<alias>.yaml` |
| `--max-model-len` | config's `vllm.max_model_len` | Override vLLM context length |
| `--workers` | `4` | Tasks run in parallel |

Re-grade an existing run (writes its `report.json`):

```bash
python scripts/grade_run.py runs/<run-dir>
```

## Per-model config (`configs/<alias>.yaml`)

```yaml
vllm:
  max_model_len: 40960        # read by run_eval.py
agent:
  step_limit: 50
model:
  model_kwargs:
    temperature: 0.7
    top_p: 0.8
    max_tokens: 2048
    extra_body:               # vLLM-only params: top_k, min_p, chat_template_kwargs
      top_k: 20
```

## Output: `runs/<alias>_<timestamp>/`

`preds.json` (patches) · `<task>/<task>.traj.json` (full agent transcript) ·
`vllm.log` · `report.json` (exit statuses, submitted / empty / graded, resolved, pass rates)

## Setup notes

- Grading needs swebench's Modal backend patched after every swebench install:
  `python scripts/patch_swebench_modal.py`
- Modal login: `~/.modal.toml` (`modal setup`). Agent sandboxes appear under the `swe-rex` app.
