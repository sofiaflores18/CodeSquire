from trl import SFTTrainer, SFTConfig
import torch
from data_preprocessing import data_preprocessing_SFT, DEFAULT_MAX_LENGTH, DEFAULT_EVAL_FRACTION
import click


class CodeSquireTrainer:
    def __init__(self):
        self.trainer = None

    def build(self, model_name, num_samples, max_length, eval_fraction):
        training_args = SFTConfig(
            output_dir="outputs",
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
            # Requires a chat template with {% generation %}; Qwen3's has it.
            assistant_only_loss=True,
            learning_rate=1e-5,
            num_train_epochs=2,
            eval_strategy="steps",
            eval_steps=25,
            logging_steps=5,
            save_strategy="steps",
            save_steps=50,
            report_to="none",
        )
        splits = data_preprocessing_SFT(
            num_samples,
            model_name=model_name,
            max_length=max_length,
            eval_fraction=eval_fraction,
        )
        self.trainer = SFTTrainer(
            model=model_name,
            train_dataset=splits["train"],
            eval_dataset=splits["test"],
            args=training_args,
        )

    def train(self, model_name, num_samples, max_length, eval_fraction):
        self.build(model_name, num_samples, max_length, eval_fraction)
        self.trainer.train()

    def resume_training_from_checkpoint(
        self, checkpoint_path, model_name, num_samples, max_length, eval_fraction
    ):
        # The trainer has to exist before it can resume, so build it the same way
        # a fresh run does and let the checkpoint restore the weights and state.
        self.build(model_name, num_samples, max_length, eval_fraction)
        self.trainer.train(resume_from_checkpoint=checkpoint_path)


# Command line arguments to pass in model name, number of samples to train on, and a resume flag 
@click.command()
@click.option('--model_name', default="Qwen/Qwen3-8B", help="The model you want to train with SFT")
@click.option("--resume", type=str, default=None, help="Checkpoint path")
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
