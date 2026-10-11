#!/usr/bin/env python
"""Grade one run directory and write report.json next to its preds.json.

    python scripts/grade_run.py runs/qwen3-8b_20261010T180537Z

Only nonempty submissions are sent to the SWE-bench harness (on Modal).
Empty patches, step-limit exits, loops and crashes are counted from the
trajectories and reported, but never graded: they cannot resolve anything.
"""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets" / "swebench_verified_validation_25.json"


def main(out: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    ids = manifest["instance_ids"]

    preds = json.loads((out / "preds.json").read_text())
    preds = list(preds.values()) if isinstance(preds, dict) else preds
    by_id = {p["instance_id"]: p for p in preds}

    exit_statuses = collections.Counter()
    for instance_id in ids:
        traj = out / instance_id / f"{instance_id}.traj.json"
        status = json.loads(traj.read_text())["info"].get("exit_status") if traj.is_file() else None
        exit_statuses[status or "NoTrajectory"] += 1

    submitted = [i for i in ids if i in by_id]
    to_grade = [i for i in submitted if (by_id[i].get("model_patch") or "").strip()]

    graded = {}
    if to_grade:
        # The harness writes logs/ and <model>.<run_id>.json into its cwd.
        run_id = out.name
        subprocess.run(
            [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "--dataset_name", manifest["dataset"],
                "--split", manifest["split"],
                "--predictions_path", str(out / "preds.json"),
                "--instance_ids", *to_grade,
                "--run_id", run_id,
                "--timeout", "600",
                "--modal", "true",
            ],
            cwd=out,
            check=True,
        )
        graded = json.loads(next(out.glob(f"*.{run_id}.json")).read_text())

    resolved = graded.get("resolved_ids", [])
    report = {
        "run": out.name,
        "model": preds[0]["model_name_or_path"] if preds else None,
        "tasks": len(ids),
        "exit_statuses": dict(exit_statuses.most_common()),
        "submitted": len(submitted),
        "empty_submissions": len(submitted) - len(to_grade),
        "graded": len(to_grade),
        "resolved": len(resolved),
        "unresolved": len(graded.get("unresolved_ids", [])),
        "grading_errors": len(graded.get("error_ids", [])),
        "pass_rate_graded": round(len(resolved) / len(to_grade), 4) if to_grade else None,
        "pass_rate_all_tasks": round(len(resolved) / len(ids), 4),
        "resolved_ids": resolved,
        "unresolved_ids": graded.get("unresolved_ids", []),
        "grading_error_ids": graded.get("error_ids", []),
        "empty_submission_ids": [i for i in submitted if i not in to_grade],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{report['model']} ({out.name})")
    print("  exit statuses:  " + ", ".join(f"{k} {v}" for k, v in report["exit_statuses"].items()))
    print(f"  submitted {report['submitted']}/{len(ids)} | empty {report['empty_submissions']} "
          f"| graded {report['graded']} | grading errors {report['grading_errors']}")
    rate = f"{report['pass_rate_graded']:.0%}" if to_grade else "n/a"
    print(f"  resolved {report['resolved']}/{report['graded']} graded ({rate}) "
          f"= {report['resolved']}/{len(ids)} tasks ({report['pass_rate_all_tasks']:.0%})")
    print(f"  report: {out / 'report.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    main(parser.parse_args().run_dir.resolve())
