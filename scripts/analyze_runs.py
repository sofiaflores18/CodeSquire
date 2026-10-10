#!/usr/bin/env python3
"""Audit saved runs without executing model output or launching evaluation.

Usage: python scripts/analyze_runs.py --output reports/20261010
Uses only the standard library and git (read-only patch syntax inspection).
Response-span times exclude sandbox startup and unrecorded final failures; they
are not complete task latencies. Missing grades are never converted to rewards.
"""

import argparse
import collections
import csv
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


def reward_keys(value, prefix=""):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if "reward" in key.lower():
                found.append(path)
            found.extend(reward_keys(item, path))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(reward_keys(item, f"{prefix}[{i}]"))
    return found


def audit_run(run):
    status = json.loads((run / "overnight-status.json").read_text(encoding="utf-8"))
    preds = json.loads((run / "preds.json").read_text(encoding="utf-8"))
    if isinstance(preds, dict):
        preds = list(preds.values())
    predictions = {row["instance_id"]: row for row in preds}
    selected = (run / "selected_ids.txt").read_text().splitlines()
    assert len(preds) == len(predictions) == len(selected) == len(set(selected))
    assert set(selected) == set(predictions)
    rows, patch_checks, all_keys, all_times = [], [], [], []
    for path in sorted(run.glob("*/*.traj.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        info = data["info"]
        iid = data["instance_id"]
        patch = predictions[iid].get("model_patch") or ""
        assert patch == (info.get("submission") or ""), (run.name, iid)
        responses, tools, accepted, errors, times, api_latencies = [], [], [], [], [], []
        for message in data["messages"]:
            extra = message.get("extra", {})
            response = extra.get("response")
            if response:
                responses.append(response)
                if response.get("created"):
                    times.append(response["created"])
                if extra.get("timestamp") and response.get("created"):
                    api_latencies.append(extra["timestamp"] - response["created"])
            if extra.get("timestamp"):
                times.append(extra["timestamp"])
            if message["role"] == "assistant":
                accepted.append(message)
            if message["role"] == "tool":
                tools.append(message)
            if extra.get("interrupt_type") == "FormatError":
                errors.append(message)
        all_times.extend(times)
        all_keys.extend(reward_keys(data))
        if patch.strip():
            check = subprocess.run(
                ["git", "apply", "--numstat", "-"], input=patch,
                text=True, capture_output=True, check=False,
            )
            patch_checks.append({"instance_id": iid, "syntax_exit_code": check.returncode,
                                 "numstat": check.stdout.strip(), "error": check.stderr.strip()})
        rows.append({
            "model": status["model_alias"], "instance_id": iid,
            "exit_status": info["exit_status"], "nonempty_patch": bool(patch.strip()),
            "patch_characters": len(patch), "api_calls": info["model_stats"]["api_calls"],
            "saved_responses": len(responses), "accepted_assistant_turns": len(accepted),
            "accepted_actions": sum(len(m.get("extra", {}).get("actions", [])) for m in accepted),
            "tool_observations": len(tools),
            "nonzero_tool_observations": sum(m["extra"].get("returncode", 0) != 0 for m in tools),
            "format_errors": len(errors),
            "unknown_tool_errors": sum("Unknown tool" in m["content"] for m in errors),
            "length_finishes": sum(r["choices"][0]["finish_reason"] == "length" for r in responses),
            "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in responses),
            "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in responses),
            "format_error_completion_tokens": sum(m["extra"]["response"].get("usage", {}).get("completion_tokens", 0) for m in errors),
            "response_span_seconds": round(max(times) - min(times), 3) if times else None,
            "accepted_response_latency_median_seconds": round(statistics.median(api_latencies), 3) if api_latencies else None,
            "large_output_observations": sum(len(m["extra"].get("raw_output", "")) >= 10000 for m in tools),
            "dependency_error_observations": sum(bool(re.search(r"^(?:ModuleNotFoundError|ImportError):", m["extra"].get("raw_output", ""), re.MULTILINE)) for m in tools),
            "official_reward": None,
        })
    assert {row["instance_id"] for row in rows} == set(selected)
    counts = collections.Counter(row["exit_status"] for row in rows)
    totals = {key: sum(row[key] for row in rows) for key in (
        "api_calls", "saved_responses", "accepted_assistant_turns", "accepted_actions",
        "tool_observations", "nonzero_tool_observations", "format_errors", "unknown_tool_errors",
        "length_finishes", "prompt_tokens", "completion_tokens", "format_error_completion_tokens",
        "large_output_observations", "dependency_error_observations",
    )}
    elapsed = status["elapsed_seconds"]
    spans = sorted(row["response_span_seconds"] for row in rows if row["response_span_seconds"] is not None)
    summary = {
        "model": status["model_alias"], "hf_model": status["hf_model"], "run_id": run.name,
        "selected_tasks": len(selected), "runner_exit_code": status["runner_exit_code"],
        "started_utc": status["started_utc"], "finished_utc": status["finished_utc"],
        "elapsed_seconds": elapsed, "amortized_seconds_per_attempt": elapsed / len(rows),
        "attempts_per_hour_including_overhead": round(len(rows) * 3600 / elapsed, 2),
        "observed_response_window_seconds": round(max(all_times) - min(all_times), 3),
        "response_span_median_seconds": statistics.median(spans),
        "response_span_p90_seconds": spans[max(0, math.ceil(.9 * len(spans)) - 1)],
        "api_calls_per_task_median": statistics.median(row["api_calls"] for row in rows),
        "api_calls_per_task_max": max(row["api_calls"] for row in rows),
        "exit_statuses": dict(counts), "nonempty_patches": sum(row["nonempty_patch"] for row in rows),
        "tasks_with_tool_observations": sum(row["tool_observations"] > 0 for row in rows),
        "tasks_with_dependency_errors": sum(row["dependency_error_observations"] > 0 for row in rows),
        "reward_field_paths": sorted(set(all_keys)),
        "grading_report_files": [str(p.relative_to(run)) for p in run.rglob("*.json") if p.name in ("results.json", "report.json")],
        "patch_syntax_checks": patch_checks, **totals,
    }
    assert len(rows) == status["prediction_stats"]["predictions"]
    assert summary["nonempty_patches"] == status["prediction_stats"]["nonempty_patches"]
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="Also write PNG/PDF charts (requires matplotlib)")
    args = parser.parse_args()
    summaries, rows = [], []
    reference_ids = None
    for run in sorted(args.runs.iterdir()):
        if run.is_dir() and (run / "overnight-status.json").exists():
            summary, instances = audit_run(run)
            summaries.append(summary)
            rows.extend(instances)
            instance_ids = {row["instance_id"] for row in instances}
            if reference_ids is None:
                reference_ids = instance_ids
            assert instance_ids == reference_ids, "Runs use different task selections"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    if rows:
        with (args.output / "tasks.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_matrix(summaries, rows, args.output)
    if args.plot:
        plot_summary(summaries, args.output)
    print(f"Audited {len(summaries)} runs, {len(rows)} attempts; wrote {args.output}")


def write_matrix(summaries, rows, output):
    models = [row["model"] for row in sorted(summaries, key=lambda row: row["started_utc"])]
    lookup = {(row["instance_id"], row["model"]): row for row in rows}
    lines = [
        "**Outcome for each selected task.**", "",
        "C = context overflow; F = repeated format error; E = submitted empty; "
        "P = nonempty submission, ungraded. P does not mean resolved. "
        "All models have the same task IDs.", "",
        "| Instance | " + " | ".join(models) + " |",
        "|" + "---|" * (len(models) + 1),
    ]
    for iid in sorted({row["instance_id"] for row in rows}):
        cells = []
        for model in models:
            row = lookup[iid, model]
            cells.append({
                "ContextWindowExceededError": "C", "RepeatedFormatError": "F",
                "Submitted": "P" if row["nonempty_patch"] else "E",
            }.get(row["exit_status"], row["exit_status"]))
        lines.append("| " + iid + " | " + " | ".join(cells) + " |")
    (output / "task_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(summaries, key=lambda row: row["started_utc"])
    labels = [row["model"] for row in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.suptitle("CodeSquire: 25 attempts per model, 4 concurrent workers", fontsize=15, fontweight="bold")
    minutes = [row["elapsed_seconds"] / 60 for row in ordered]
    axes[0].barh(labels, minutes, color="#346c9c")
    for i, value in enumerate(minutes):
        axes[0].text(value + .2, i, f"{value:.2f}", va="center", fontsize=10)
    axes[0].set_xlim(0, max(minutes) * 1.22)
    axes[0].set_xlabel("Whole-run elapsed minutes (includes overhead)")
    axes[0].set_title("Runtime of these failed runs")
    categories = [
        ("Repeated format error", "#b85b56", lambda r: r["exit_statuses"].get("RepeatedFormatError", 0)),
        ("Context overflow", "#d9ab50", lambda r: r["exit_statuses"].get("ContextWindowExceededError", 0)),
        ("Submitted empty", "#a8adb5", lambda r: r["exit_statuses"].get("Submitted", 0) - r["nonempty_patches"]),
        ("Submitted nonempty (ungraded)", "#478a84", lambda r: r["nonempty_patches"]),
    ]
    left = [0] * len(ordered)
    for label, color, count in categories:
        values = [count(row) for row in ordered]
        axes[1].barh(labels, values, left=left, color=color, label=label)
        for i, value in enumerate(values):
            if value:
                axes[1].text(left[i] + value / 2, i, str(value), ha="center", va="center", fontsize=10)
        left = [x + y for x, y in zip(left, values)]
    axes[1].set_title("Agent termination, not benchmark correctness")
    axes[1].set_xlabel("Attempts")
    axes[1].set_xlim(0, 25)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2, frameon=False, fontsize=9)
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .025, "No official grading reports are present. Faster termination does not imply better solving throughput.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .93))
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
