from trl import GRPOTrainer
from trl.rewards import accuracy_reward
import click

class CodeSquireTrainer:
    def __init__(self):
        self.trainer = None

    def train(self):
        ...

    def resume_training_from_checkpoint(self, checkpoint_path:str):
        ...

@click.command()
def main():
    trainer = CodeSquireTrainer()
    trainer.train()
    
if __name__ == "__main__":
    main()
