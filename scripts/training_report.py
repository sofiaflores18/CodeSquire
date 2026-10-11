#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "tensorboardX>=2.6", "huggingface_hub>=0.25"]
# ///
"""Turn a Trainer's trainer_state.json into TensorBoard logs, charts, and a health report.

The SFT run sets report_to="none", so no tfevents are written while training. Every
logged scalar still survives in the checkpoint's trainer_state.json, so this replays
that log history after the fact:

    scripts/training_report.py --hf-repo Adammouedden/squire        # pull from the Hub
    scripts/training_report.py outputs/checkpoint-238               # or a local checkpoint

Writes tfevents under --tb-out (point `tensorboard --logdir` at it, or upload the
directory to a HF model repo, where the Hub renders it as a Metrics tab), a PNG panel
at --png-out, and a text health summary on stdout.
"""

import argparse
import json
import statistics
from pathlib import Path

# Logged under these names by TRL/Transformers. "loss" is the train-side one; the
# eval pass prefixes everything with eval_.
TRAIN_SCALARS = ("loss", "grad_norm", "learning_rate", "entropy", "mean_token_accuracy")
EVAL_SCALARS = ("eval_loss", "eval_entropy", "eval_mean_token_accuracy")


def load_state(source: str, filename: str) -> dict:
    """Accept a local checkpoint dir, a local trainer_state.json, or a HF repo id."""
    local = Path(source)
    if local.is_dir():
        local = local / "trainer_state.json"
    if local.is_file():
        return json.loads(local.read_text())

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=source, filename=filename)
    return json.loads(Path(path).read_text())


def split_history(state: dict) -> tuple[list[dict], list[dict]]:
    history = state["log_history"]
    train = [e for e in history if "loss" in e]
    evals = [e for e in history if "eval_loss" in e]
    return train, evals


def find_restarts(train: list[dict]) -> list[int]:
    """Steps where the LR schedule jumped back up.

    save_only_model=True drops the optimizer state, so a resume rebuilds the
    scheduler from zero: the LR climbs instead of continuing to decay. That is the
    signature of a resume, and it matters for reading the curves.
    """
    steps = []
    for prev, cur in zip(train, train[1:]):
        if cur["learning_rate"] > prev["learning_rate"] * 1.05:
            steps.append(cur["step"])
    return steps


def write_tensorboard(train: list[dict], evals: list[dict], out: Path) -> Path:
    from tensorboardX import SummaryWriter

    out.mkdir(parents=True, exist_ok=True)
    # TensorBoard unions every event file in a directory, so a re-run would stack a
    # second copy of the same curves on top of the first. These are derived files we
    # wrote ourselves, so replace them.
    for stale in out.glob("events.out.tfevents.*"):
        stale.unlink()
    with SummaryWriter(logdir=str(out)) as writer:
        for entry in train:
            for key in TRAIN_SCALARS:
                if key in entry:
                    writer.add_scalar(f"train/{key}", entry[key], entry["step"])
        for entry in evals:
            for key in EVAL_SCALARS:
                if key in entry:
                    writer.add_scalar(f"eval/{key[len('eval_'):]}", entry[key], entry["step"])
            # Plotted against the train loss, this is the overfitting check.
            writer.add_scalar("eval/loss_vs_train", entry["eval_loss"], entry["step"])
    return out


def plot(train: list[dict], evals: list[dict], restarts: list[int], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # dataviz reference palette, light mode: slot 1 blue (train), slot 2 orange (eval).
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": grid,
            "axes.labelcolor": muted,
            "axes.titlecolor": ink,
            "text.color": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "grid.color": grid,
            "font.size": 9,
            "axes.titlesize": 10,
        }
    )

    steps = [e["step"] for e in train]
    eval_steps = [e["step"] for e in evals]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    def frame(ax, title):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("step")
        for step in restarts:
            # A resume is a discontinuity in the curves, so mark it rather than
            # letting it read as a training anomaly.
            ax.axvline(step, color=muted, linewidth=1, linestyle=":", alpha=0.7)
        return ax

    ax = frame(axes[0][0], "Loss — train vs held-out eval")
    ax.plot(steps, [e["loss"] for e in train], color=blue, linewidth=2, label="train")
    ax.plot(eval_steps, [e["eval_loss"] for e in evals], color=orange, linewidth=2,
            marker="o", markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=2,
            label="eval")
    ax.annotate(f"{evals[-1]['eval_loss']:.3f}", (eval_steps[-1], evals[-1]["eval_loss"]),
                textcoords="offset points", xytext=(6, -2), color=muted, fontsize=8)
    ax.legend(frameon=False, loc="upper right")

    ax = frame(axes[0][1], "Gradient norm (pre-clip)")
    ax.plot(steps, [e["grad_norm"] for e in train], color=blue, linewidth=2)
    ax.set_ylim(bottom=0)

    ax = frame(axes[1][0], "Learning rate")
    ax.plot(steps, [e["learning_rate"] for e in train], color=blue, linewidth=2)
    ax.set_ylim(bottom=0)

    ax = frame(axes[1][1], "Mean token accuracy — train vs eval")
    ax.plot(steps, [e["mean_token_accuracy"] for e in train], color=blue, linewidth=2,
            label="train")
    ax.plot(eval_steps, [e["eval_mean_token_accuracy"] for e in evals], color=orange,
            linewidth=2, marker="o", markersize=7, markeredgecolor="#fcfcfb",
            markeredgewidth=2, label="eval")
    ax.legend(frameon=False, loc="lower right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def report(state: dict, train: list[dict], evals: list[dict], restarts: list[int]) -> str:
    gnorm = [e["grad_norm"] for e in train]
    # Compare the last few train logs against the final eval, not single points:
    # the train loss is noisy at batch size 1 x 8.
    tail = statistics.mean(e["loss"] for e in train[-5:])
    final_eval = evals[-1]["eval_loss"]
    regressions = sum(1 for a, b in zip(evals, evals[1:]) if b["eval_loss"] > a["eval_loss"])
    best_is_last = state.get("best_global_step") == state.get("global_step")

    lines = [
        "Training health",
        "=" * 60,
        f"steps                {state['global_step']} / {state['max_steps']}  "
        f"({state['num_train_epochs']} epochs)",
        f"best eval_loss       {state['best_metric']:.4f} at step {state['best_global_step']}"
        + ("  (= final step)" if best_is_last else ""),
        f"eval_loss curve      {evals[0]['eval_loss']:.4f} -> {final_eval:.4f}, "
        f"{regressions} of {len(evals) - 1} steps up",
        f"train/eval gap       {tail:.4f} train (last 5 logs) vs {final_eval:.4f} eval "
        f"= {tail - final_eval:+.4f}",
        f"grad_norm            mean {statistics.mean(gnorm):.2f}, max {max(gnorm):.2f}, "
        f"no NaN/inf" if all(g == g and g != float("inf") for g in gnorm) else "",
        f"token accuracy       {train[0]['mean_token_accuracy']:.3f} -> "
        f"{evals[-1]['eval_mean_token_accuracy']:.3f} (eval)",
        f"entropy              {train[0]['entropy']:.3f} -> {evals[-1]['eval_entropy']:.3f} "
        f"(eval)",
        f"tokens seen          {train[-1]['num_tokens'] / 1e6:.1f}M in the last segment",
        f"total_flos           {state['total_flos']:.3e}",
    ]
    if restarts:
        lines += [
            "",
            f"LR schedule restarted at step(s) {restarts}: the run was resumed, and "
            "save_only_model=True",
            "drops the optimizer state, so the scheduler and Adam moments restart from "
            "scratch there.",
            "The LR therefore never annealed to zero — it ended at "
            f"{train[-1]['learning_rate']:.2e}.",
        ]
    return "\n".join(line for line in lines if line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="checkpoint dir or trainer_state.json path")
    parser.add_argument("--hf-repo", help="HF model repo to pull trainer_state.json from")
    parser.add_argument("--hf-file", default="checkpoint-238/trainer_state.json",
                        help="path to trainer_state.json inside --hf-repo")
    parser.add_argument("--tb-out", type=Path, default=Path("runs/tb/sft"),
                        help="directory for the tfevents file")
    parser.add_argument("--png-out", type=Path, default=Path("runs/tb/training_health.png"))
    args = parser.parse_args()

    source = args.source or args.hf_repo
    if not source:
        parser.error("give a local checkpoint path or --hf-repo")

    state = load_state(source, args.hf_file)
    train, evals = split_history(state)
    if not train:
        parser.error("no train-side entries in log_history")
    restarts = find_restarts(train)

    tb = write_tensorboard(train, evals, args.tb_out)
    png = plot(train, evals, restarts, args.png_out)
    print(report(state, train, evals, restarts))
    print(f"\ntensorboard --logdir {tb}")
    print(f"chart: {png}")


if __name__ == "__main__":
    main()
