
import argparse
import os
import subprocess
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent
PORT = 8000

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

with open(ROOT / "models.yaml") as f:
    models = yaml.safe_load(f)["models"]

hf_id = models[args.model]

env = os.environ.copy()
env["OPENAI_API_KEY"] = "EMPTY"
env["OPENAI_API_BASE"] = f"http://localhost:{PORT}/v1"
env["MSWEA_COST_TRACKING"] = "ignore_errors"

# Start vLLM
server = subprocess.Popen(
    [
        "vllm", "serve", hf_id,
        "--dtype", "half",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.85",
        "--port", str(PORT),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ],
    cwd=ROOT,
    env=env,
)

try:
    # Wait until the requested model is available
    print(f"Starting vLLM: {hf_id}", flush=True)

    for _ in range(600):
        if server.poll() is not None:
            raise RuntimeError("vLLM exited during startup")

        try:
            response = requests.get(
                f"http://localhost:{PORT}/v1/models",
                timeout=2,
            )
            if response.ok:
                available = [
                    item["id"]
                    for item in response.json()["data"]
                ]
                if hf_id in available:
                    break
        except requests.RequestException:
            pass

        time.sleep(2)
    else:
        raise TimeoutError("vLLM did not start within 20 minutes")

    # Generate the model-specific mini-SWE-agent config
    config = ROOT / "configs" / f"{args.model}.yaml"
    config.parent.mkdir(exist_ok=True)

    config.write_text(
        yaml.safe_dump({
            "model": {
                "model_name": f"openai/{hf_id}",
                "model_kwargs": {
                    "tool_choice": "required",
                },
            },
        })
    )

    # Run the same 25 SWE-bench tasks and grade the patches
    run_id = f"{args.model}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"

    subprocess.run(
        [
            "bash", "scripts/run_swebench.sh", "all",
            "--config", str(config.relative_to(ROOT)),
            "--run-id", run_id,
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

finally:
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
