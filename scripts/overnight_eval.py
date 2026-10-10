#!/usr/bin/env python3
"""Run Docker-free CodeSquire SWE-bench evaluations sequentially and push results.

Place at CodeSquire/scripts/overnight_eval.py alongside scripts/remote_eval.sh.
Only artifacts in each runs/<run_id>/ folder are committed; the cloned worktree,
model weights, and credential-like files are excluded.
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "remote_eval.sh"
MODELS_FILE = ROOT / "models.yaml"
PORT = 8000
MAX_GIT_FILE_BYTES = 90 * 1024 * 1024  # GitHub rejects files >= 100 MB.


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def print_status(message):
    print(f"[{now()}] {message}", flush=True)


def git(*args, check=True, quiet=False):
    result = subprocess.run(
        ["git", *args], cwd=ROOT,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        capture_output=True, text=True,
    )
    if not quiet and result.returncode:
        print_status(f"git {' '.join(args[:2])} failed: {(result.stderr or result.stdout).strip()}")
    if check and result.returncode:
        raise RuntimeError(f"Git command failed: git {' '.join(args)}")
    return result


def check_port_available(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def get_json(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def wait_for_model(process, model_id, port, timeout):
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}. Check its log.")
        try:
            names = [entry.get("id") for entry in get_json(url)["data"]]
            if model_id in names:
                return
        except (OSError, ValueError, KeyError, TypeError):
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM did not serve {model_id} within {timeout}s")


def probe_tool_call(model_id, port):
    # A quick sanity check. This does not prove the agent will use tools well.
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Call bash to execute echo hello."}],
        "tools": [{"type": "function", "function": {
            "name": "bash", "description": "Execute bash commands",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]},
        }}],
        "tool_choice": "required", "max_tokens": 256, "temperature": 0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    return any(
        call.get("function", {}).get("name") == "bash"
        for call in result["choices"][0]["message"].get("tool_calls") or []
    )


def stop_process_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def prediction_stats(path):
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        rows = list(data.values()) if isinstance(data, dict) else data
        return {
            "predictions": len(rows),
            "nonempty_patches": sum(bool((r.get("model_patch") or "").strip()) for r in rows),
        }
    except (ValueError, TypeError, AttributeError):
        return {"error": "Could not parse preds.json"}


def result_files(output_dir, include_logs):
    """Publish experiment artifacts only; never publish cloned repos or binary weights."""
    omitted = []
    files = []
    if not output_dir.exists():
        return files, omitted
    for current, dirs, names in os.walk(output_dir, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {
            "worktree", ".git", ".venv", "__pycache__", "node_modules", "secrets"
        }]
        for name in names:
            path = Path(current) / name
            rel = path.relative_to(ROOT)
            low = name.lower()
            if path.is_symlink() or not path.is_file():
                omitted.append(str(rel))
                continue
            if (low == "model-config.yaml" or low.startswith(".env") or "token" in low or low.endswith((
                    ".key", ".pem", ".safetensors", ".pt", ".pth", ".bin"))):
                omitted.append(str(rel))
                continue
            if low.endswith(".log") and not include_logs:
                omitted.append(str(rel))
                continue
            if path.stat().st_size > MAX_GIT_FILE_BYTES:
                omitted.append(str(rel) + " (over GitHub file limit)")
                continue
            files.append(str(rel))
    return files, omitted


def push_results(output_dir, alias, run_id, include_logs, branch):
    files, omitted = result_files(output_dir, include_logs)
    if omitted:
        print_status(f"Excluded {len(omitted)} artifact(s) from GitHub (see publish-manifest.txt).")
        manifest = output_dir / "publish-manifest.txt"
        manifest.write_text("Excluded from GitHub:\n" + "\n".join(omitted) + "\n")
        files.append(str(manifest.relative_to(ROOT)))
    if not files:
        print_status("No artifacts to push for this model.")
        return False

    # -f handles repositories with a `runs/` gitignore entry. Commit ONLY these
    # experiment artifacts, not teammates' staged/unstaged source changes.
    git("add", "-f", "--", *files)
    result = git("commit", "--only", "-m", f"eval: {alias} SWE-bench 25 ({run_id})", "--", *files,
                 check=False, quiet=True)
    if result.returncode != 0:
        if "nothing to commit" in (result.stdout + result.stderr).lower():
            print_status("Results already committed; attempting push.")
        else:
            print_status(f"Commit failed: {(result.stderr or result.stdout).strip()}")
            return False
    for retry in range(1, 4):
        outcome = git("push", "origin", f"HEAD:refs/heads/{branch}", check=False, quiet=True)
        if outcome.returncode == 0:
            print_status(f"GitHub push succeeded: branch {branch}, run {run_id}")
            return True
        print_status(f"GitHub push attempt {retry}/3 failed: {(outcome.stderr or outcome.stdout).strip()}")
        if retry < 3:
            time.sleep(10)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", help="Model aliases from models.yaml (default: all)")
    p.add_argument("--mode", choices=["infer", "all"], default="all",
                   help="all = infer + cloud grading; infer = predictions only")
    p.add_argument("--hours-per-model", type=float, default=3,
                   help="Time limit for inference+cloud grading per model (0 disables limit)")
    p.add_argument("--startup-minutes", type=float, default=30,
                   help="Max minutes for model download and vLLM startup")
    p.add_argument("--include-logs", action="store_true",
                   help="Also commit raw console logs (only for a trusted/private repository)")
    p.add_argument("--skip-tool-probe", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print planned models without launching anything")
    args = p.parse_args()

    if not RUNNER.is_file() or not MODELS_FILE.is_file():
        p.error(f"Expected {RUNNER} and {MODELS_FILE}")
    models = yaml.safe_load(MODELS_FILE.read_text())["models"]
    selected = args.models or list(models)
    for name in selected:
        if name not in models or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            p.error(f"Unknown/invalid model alias: {name}")
        if not isinstance(models[name], str):
            p.error(f"Expected models.{name} to be a Hugging Face ID string")
    print_status("Queue: " + ", ".join(f"{x}={models[x]}" for x in selected))
    if args.dry_run:
        return 0
    if not shutil.which("vllm") or not shutil.which("mini-extra"):
        p.error("Activate your Python environment with vllm and mini-extra installed")
    if not check_port_available(PORT):
        p.error("Port 8000 is occupied. Stop the currently running vLLM server first.")
    if args.mode == "all" and not shutil.which("sb-cli"):
        p.error("Cloud grading requires sb-cli; install it or use --mode infer")
    if args.mode == "all" and not os.environ.get("SWEBENCH_API_KEY"):
        p.error("Set SWEBENCH_API_KEY for cloud grading, or use --mode infer")

    if git("rev-parse", "--is-inside-work-tree", quiet=True).stdout.strip() != "true":
        p.error("CodeSquire must be a git repository")
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD", quiet=True).stdout.strip()
    if not branch:
        p.error("Check out a branch before running; HEAD is detached")
    if git("remote", "get-url", "origin", check=False, quiet=True).returncode:
        p.error("Git remote 'origin' is missing")
    for field in ("user.name", "user.email"):
        if not git("config", field, check=False, quiet=True).stdout.strip():
            p.error(f"Configure git {field} before running")
    # A dry-run push confirms credentials without changing GitHub.
    try_push = git("push", "--dry-run", "origin", f"HEAD:refs/heads/{branch}", check=False, quiet=True)
    if try_push.returncode:
        p.error("GitHub push preflight failed: " + (try_push.stderr or try_push.stdout).strip())

    batch_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = ROOT / "overnight-logs" / batch_id
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OPENAI_API_BASE": f"http://127.0.0.1:{PORT}/v1",
        "OPENAI_API_KEY": env.get("OPENAI_API_KEY") or "EMPTY",
        "MSWEA_COST_TRACKING": "ignore_errors",
        "GIT_TERMINAL_PROMPT": "0",
    })
    failures = []
    for alias in selected:
        model_id = models[alias]
        run_id = f"{batch_id}_{alias}"
        out = ROOT / "runs" / run_id
        vllm_log = log_dir / f"{alias}.vllm.log"
        infer_log = log_dir / f"{alias}.runner.log"
        process = None
        evaluation = None
        status = {
            "model_alias": alias, "hf_model": model_id,
            "run_id": run_id, "started_utc": now(), "mode": args.mode,
            "runner_exit_code": None, "error": None,
        }
        print_status(f"=== Starting {alias} ({model_id}) ===")
        begin = time.monotonic()
        try:
            if not check_port_available(PORT):
                raise RuntimeError("Port 8000 is occupied by another process")
            # Hermes works with Qwen 2.5/3. Other checkpoints may need a different
            # chat template/parser. A failed model does not stop the queue.
            command = [
                "vllm", "serve", model_id, "--host", "127.0.0.1",
                "--port", str(PORT), "--dtype", "half", "--max-model-len", "8192",
                "--gpu-memory-utilization", "0.85",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
            ]
            with vllm_log.open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            wait_for_model(process, model_id, PORT, args.startup_minutes * 60)
            print_status(f"vLLM ready: {model_id}")
            if not args.skip_tool_probe:
                try:
                    if not probe_tool_call(model_id, PORT):
                        print_status("WARNING: bash tool-call probe returned no structured tool call")
                    else:
                        print_status("Structured bash tool call: OK")
                except Exception as error:
                    print_status(f"WARNING: tool-call probe failed: {error}")
            # The runner creates runs/<run_id> itself; DO NOT create it here.
            command = ["bash", "scripts/remote_eval.sh", args.mode,
                       "--model", alias, "--run-id", run_id,
                       "--output", f"runs/{run_id}"]
            print_status("Running: " + " ".join(command))
            with infer_log.open("w") as log:
                evaluation = subprocess.Popen(command, cwd=ROOT, env=env,
                                              stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                try:
                    timeout = None if args.hours_per_model == 0 else args.hours_per_model * 3600
                    status["runner_exit_code"] = evaluation.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    status["error"] = f"Exceeded {args.hours_per_model} hours per model"
                    stop_process_group(evaluation)
                    status["runner_exit_code"] = -1
            if status["runner_exit_code"]:
                print_status(f"Runner failed for {alias}; see {infer_log}")
        except Exception as error:
            status["error"] = str(error)
            print_status(f"Failure for {alias}: {error}")
        finally:
            stop_process_group(evaluation)
            stop_process_group(process)
            # Allow vLLM's GPU worker processes time to release resources.
            time.sleep(3)

        out.mkdir(parents=True, exist_ok=True)
        status.update({"finished_utc": now(),
                       "elapsed_seconds": round(time.monotonic() - begin),
                       "prediction_stats": prediction_stats(out / "preds.json")})
        (out / "overnight-status.json").write_text(json.dumps(status, indent=2) + "\n")
        try:
            pushed = push_results(out, alias, run_id, args.include_logs, branch)
        except Exception as error:
            pushed = False
            print_status(f"GitHub publishing failed: {error}")
        if not pushed or status["runner_exit_code"] != 0:
            failures.append(alias)
        print_status(f"=== Finished {alias}; artifacts: {out} ===")

    print_status(f"All requested models attempted. Failed/publish-incomplete: {', '.join(failures) if failures else 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_status("Interrupted")
        sys.exit(130)
