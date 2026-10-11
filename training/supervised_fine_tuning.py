from trl import SFTTrainer
from datasets import load_dataset
from data_preprocessing import dataset

trainer = SFTTrainer(model="WeiboAI/VibeThinker-1.5B", train_dataset=dataset)
trainer.train()