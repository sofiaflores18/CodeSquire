from trl import SFTTrainer
from datasets import load_dataset
from data_preprocessing import data_preprocessing_SFT
import click

@click.command()
@click.option('--model_name', help="The model you want to train with SFT")
@click.option('--num_samples', type=int)
def main(model_name="Qwen/Qwen3-8B", num_samples=100):

    dataset = data_preprocessing_SFT(num_samples)

    trainer = SFTTrainer(model=model_name, train_dataset=dataset)

    trainer.train()

if __name__ == "__main__":
    main()