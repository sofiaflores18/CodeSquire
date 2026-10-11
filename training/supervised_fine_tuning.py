from trl import SFTTrainer
from datasets import load_dataset
from data_preprocessing import qwen_coder_trajectories_subdataset

trainer = SFTTrainer(model="WeiboAI/VibeThinker-1.5B", train_dataset=qwen_coder_trajectories_subdataset)
trainer.train()