from trl import SFTTrainer
from datasets import load_dataset
from data_preprocessing import data_preprocessing_SFT
import click
from transformers import TrainingArguments


class CodeSquireTrainer:
    def __init__(self):
        self.trainer = None

    def train(self, model_name, num_samples):
        
        training_args = TrainingArguments(output_dir="outputs",save_strategy="steps",save_steps=50)
        dataset = data_preprocessing_SFT(num_samples)
        self.trainer = SFTTrainer(model=model_name, train_dataset=dataset, args=training_args)
        self.trainer.train()

    def resume_training_from_checkpoint(self, checkpoint_path: str):
        self.trainer.train(resume_from_checkpoint=checkpoint_path)


# Command line arguments to pass in model name, number of samples to train on, and a resume flag 
@click.command()
@click.option('--model_name', default="Qwen/Qwen3-8B", help="The model you want to train with SFT")
@click.option("--resume", type=bool, default=None, help="Checkpoint path")
@click.option('--num_samples', type=int, default=100)
def main(resume, model_name, num_samples):
    trainer = CodeSquireTrainer()

    if resume:
        trainer.resume_training_from_checkpoint(resume)

    else:
        trainer.train(model_name, num_samples)

if __name__ == "__main__":
    main()
    