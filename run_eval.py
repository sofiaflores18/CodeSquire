#!/usr/bin/env python
"""Run the 25-task SWE-bench Verified validation set against one model.

    python run_eval.py --model qwen3-8b   # uses configs/qwen3-8b.yaml
    python run_eval.py --model qwen-4b    # uses configs/qwen-4b.yaml

Starts vLLM, runs mini-SWE-agent on Modal, grades the nonempty patches with the
official SWE-bench harness on Modal, then shuts vLLM down. Everything for
the run ends up in runs/<run-id>/.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal
import requests
import yaml

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "datasets" / "swebench_verified_validation_25.json"
PORT = 8000


def kill_agent_sandboxes(label):
    """Terminate every sandbox in the swe-rex Modal app (agent environments).

    Only one eval runs at a time on this VM, so anything still alive at the
    start or end of a run is a leftover that would bill until its timeout.
    """
    try:
        app = modal.App.lookup("swe-rex")
        sandboxes = list(modal.Sandbox.list(app_id=app.app_id))
        for sandbox in sandboxes:
            sandbox.terminate()
        print(f"[{label}] terminated {len(sandboxes)} swe-rex sandboxes", flush=True)
    except Exception as error:  # cleanup must never break a run
        print(f"[{label}] sandbox cleanup failed: {error}", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="models.yaml alias")
parser.add_argument("--max-model-len", type=int, help="override the config's vllm.max_model_len")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

models = yaml.safe_load((ROOT / "models.yaml").read_text())["models"]
hf_id = models[args.model]
model_config = ROOT / "configs" / f"{args.model}.yaml"
if not model_config.is_file():
    raise SystemExit(f"Missing per-model config: {model_config}")
run_id = f"{args.model}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
out = ROOT / "runs" / run_id
out.mkdir(parents=True)

manifest = json.loads(MANIFEST.read_text())
ids = manifest["instance_ids"]
(out / "selected_ids.txt").write_text("\n".join(ids) + "\n")

# The per-model config's vllm section is for us; the rest goes to mini-SWE-agent.
agent_config = yaml.safe_load(model_config.read_text())
max_model_len = args.max_model_len or agent_config.get("vllm", {}).get("max_model_len", 32768)
agent_config.pop("vllm", None)
(out / "model-config.yaml").write_text(model_config.read_text())
model_config = out / "agent-config.yaml"
model_config.write_text(yaml.safe_dump(agent_config))

# Run plumbing, layered as swebench.yaml -> this -> agent-config.yaml.
config = out / "config.yaml"
config.write_text(yaml.safe_dump({
    "model": {
        "model_name": f"openai/{hf_id}",
        "cost_tracking": "ignore_errors",
        "model_kwargs": {
            "api_base": f"http://localhost:{PORT}/v1",
            "api_key": "local",
        },
    },
    "environment": {
        "environment_class": "swerex_modal",
        "cwd": "/testbed",
        "timeout": 600,
        # On Modal each command runs via /bin/sh with exactly this env, so the
        # image's `conda activate testbed` (BASH_ENV) never fires. Put the
        # testbed env first on PATH, or python lacks the repo's dependencies.
        "env": {
            "PATH": "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
                    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/root",
            "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R",
            "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1",
        },
        "startup_timeout": 600.0,
        # Hard cap on a sandbox's life in case cleanup is skipped (e.g. kill -9).
        "runtime_timeout": 3600.0,
        "deployment_timeout": 3600.0,
    },
    "run": {"env_startup_command": ""},
}))

print(f"Starting vLLM: {hf_id} (max-model-len {max_model_len})", flush=True)
vllm_log = open(out / "vllm.log", "w")
server = subprocess.Popen(
    [
        "vllm", "serve", hf_id,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--dtype", "half",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.85",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ],
    stdout=vllm_log,
    stderr=subprocess.STDOUT,
)

kill_agent_sandboxes("start")
try:
    for _ in range(600):
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup; see {out / 'vllm.log'}")
        try:
            r = requests.get(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.ok and hf_id in [m["id"] for m in r.json()["data"]]:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise TimeoutError("vLLM did not start within 20 minutes")

    # Inference: one attempt per task, saved to out/preds.json.
    subprocess.run(
        [
            # mini-extra swebench + per-task sandbox cleanup and a loop guard.
            sys.executable, str(ROOT / "scripts" / "swebench_runner.py"),
            "--subset", manifest["dataset"],
            "--split", manifest["split"],
            "--filter", "^(?:" + "|".join(map(re.escape, ids)) + ")$",
            "--workers", str(args.workers),
            "-c", "swebench.yaml",
            "-c", str(config),
            "-c", str(model_config),
            "--output", str(out),
        ],
        cwd=ROOT,
        env={**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"},
        check=True,
    )
finally:
    kill_agent_sandboxes("end")
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
    vllm_log.close()

# Grade only the nonempty submissions and write runs/<run-id>/report.json.
subprocess.run([sys.executable, str(ROOT / "scripts" / "grade_run.py"), str(out)], check=True)
