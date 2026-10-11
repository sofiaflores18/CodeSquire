from trl import SFTTrainer, SFTConfig
import torch
from data_preprocessing import data_preprocessing_SFT, DEFAULT_MAX_LENGTH, DEFAULT_EVAL_FRACTION
import click
import json
import zipfile
from pathlib import Path

OUTPUT_DIR = "outputs"


def checkpoint_problem(checkpoint: Path) -> str | None:
    """Return None if this checkpoint can be resumed from, else why it cannot.

    A run killed by a full disk leaves a half-written checkpoint. The weights are
    written before the optimizer state, so the usual wreckage is good weights plus
    a truncated optimizer.pt whose zip end-of-central-directory record is missing.
    torch.load dies on that, so catch it here instead of 20 minutes into a resume.
    """
    if not (checkpoint / "trainer_state.json").is_file():
        return "no trainer_state.json, so the write never finished"
    weights = list(checkpoint.glob("*.safetensors")) + list(checkpoint.glob("pytorch_model*.bin"))
    if not weights:
        return "no model weights"
    optimizer = checkpoint / "optimizer.pt"
    # is_zipfile only reads the archive's tail, so this stays cheap on a 60GB file.
    if optimizer.is_file() and not zipfile.is_zipfile(optimizer):
        return f"{optimizer.name} is truncated (disk full?); `rm` it to resume weights-only"
    return None


def resolve_checkpoint(path: str) -> Path:
    """Accept a specific checkpoint, an output directory, or "latest".

    Given a directory of checkpoints, pick the newest healthy one and say which
    unhealthy ones were skipped, rather than failing on a corrupt newest.
    """
    target = Path(OUTPUT_DIR if path == "latest" else path)
    if not target.is_dir():
        raise ValueError(f"No such checkpoint or directory: {target}")

    # A specific checkpoint was named: use it or explain why it is unusable.
    if (target / "trainer_state.json").is_file():
        problem = checkpoint_problem(target)
        if problem:
            raise ValueError(f"Cannot resume from {target}: {problem}")
        return target

    candidates = sorted(
        (d for d in target.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-")[-1]),
        reverse=True,
    )
    if not candidates:
        raise ValueError(f"No checkpoint-* directories under {target}")

    for candidate in candidates:
        problem = checkpoint_problem(candidate)
        if problem is None:
            return candidate
        print(f"Skipping {candidate.name}: {problem}")
    raise ValueError(
        f"All {len(candidates)} checkpoints under {target} are unusable; see the reasons above."
    )


def warn_if_split_changed(checkpoint: Path, num_train_examples: int, args: SFTConfig) -> None:
    """Resuming only lines up if the data split is the one the checkpoint trained on.

    The split is seeded, so it is reproducible as long as --num_samples and
    --max_length match. Back the original example count out of the saved step
    math and complain if it disagrees, because otherwise a resume with different
    flags silently trains on a different split from the same step counter.
    """
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    max_steps, epochs = state.get("max_steps"), state.get("num_train_epochs")
    if not max_steps or not epochs:
        return
    window = args.per_device_train_batch_size * args.gradient_accumulation_steps
    expected = round(max_steps / epochs) * window
    # One accumulation window of slack absorbs the trainer's own rounding.
    if abs(expected - num_train_examples) > window:
        print(
            f"WARNING: {checkpoint.name} was trained on about {expected} examples, but this "
            f"run built {num_train_examples}. Re-run with the same --num_samples and "
            f"--max_length, or the resumed steps will not match the data."
        )


class CodeSquireTrainer:
    def __init__(self):
        self.trainer = None
        self.train_examples = 0

    def build(self, model_name, num_samples, max_length, eval_fraction):
        training_args = SFTConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            bf16=True,
            model_init_kwargs={
                "dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
            },
            # Trajectories are ~9k tokens; at the old 1024 every example was cut off
            # inside the user turn, so the model never saw an assistant token.
            max_length=max_length,
            # The system and user turns are context, not targets. Masking them keeps
            # the gradient on the assistant turns instead of on the instruction
            # boilerplate, which is identical across every example in this dataset.
            # Qwen3's stock template has no {% generation %} markers, but TRL
            # swaps in a patched one that does, so the mask lands on the
            # assistant turns (verified: 14 spans for 14 assistant turns).
            assistant_only_loss=True,
            learning_rate=1e-5,
            num_train_epochs=2,
            eval_strategy="steps",
            eval_steps=25,
            logging_steps=5,
            save_strategy="steps",
            save_steps=50,
            # A full-finetune checkpoint is ~16GB of weights plus ~64GB of fp32
            # AdamW state. Keeping the optimizer state for every checkpoint
            # filled the disk mid-write and killed the run, so save weights only
            # and keep just the last two.
            save_only_model=True,
            save_total_limit=2,
            report_to="none",
        )
        splits = data_preprocessing_SFT(
            num_samples,
            model_name=model_name,
            max_length=max_length,
            eval_fraction=eval_fraction,
        )
        self.train_examples = len(splits["train"])
        self.trainer = SFTTrainer(
            model=model_name,
            train_dataset=splits["train"],
            eval_dataset=splits["test"],
            args=training_args,
        )
        return training_args

    def train(self, model_name, num_samples, max_length, eval_fraction):
        self.build(model_name, num_samples, max_length, eval_fraction)
        self.trainer.train()

    def resume_training_from_checkpoint(
        self, checkpoint_path, model_name, num_samples, max_length, eval_fraction
    ):
        # Resolve and health-check the checkpoint before paying to build the
        # trainer and tokenize the dataset, so a bad path fails in a second
        # rather than after the model is on the GPU.
        checkpoint = resolve_checkpoint(checkpoint_path)
        print(f"Resuming from {checkpoint}")

        # The trainer has to exist before it can resume, so build it the same way
        # a fresh run does and let the checkpoint restore the weights. With
        # save_only_model=True there is no optimizer state to restore, so the
        # optimizer restarts and the LR schedule is recomputed from the step count.
        training_args = self.build(model_name, num_samples, max_length, eval_fraction)
        warn_if_split_changed(checkpoint, self.train_examples, training_args)
        self.trainer.train(resume_from_checkpoint=str(checkpoint))


# Command line arguments to pass in model name, number of samples to train on, and a resume flag 
@click.command()
@click.option('--model_name', default="Qwen/Qwen3-8B", help="The model you want to train with SFT")
@click.option("--resume", type=str, default=None,
              help='Checkpoint dir, output dir, or "latest" for the newest healthy checkpoint')
@click.option('--num_samples', type=int, default=1000)
@click.option('--max_length', type=int, default=DEFAULT_MAX_LENGTH,
              help="Trajectories longer than this are dropped, not truncated")
@click.option('--eval_fraction', type=float, default=DEFAULT_EVAL_FRACTION,
              help="Fraction of the selected trajectories held out for eval")
def main(resume, model_name, num_samples, max_length, eval_fraction):
    trainer = CodeSquireTrainer()

    if resume:
        trainer.resume_training_from_checkpoint(
            resume, model_name, num_samples, max_length, eval_fraction
        )

    else:
        trainer.train(model_name, num_samples, max_length, eval_fraction)

if __name__ == "__main__":
    main()
