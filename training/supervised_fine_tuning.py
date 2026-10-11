from trl import SFTTrainer
from datasets import load_dataset
<<<<<<< HEAD
from data_preprocessing import qwen_coder_trajectories_subdataset

trainer = SFTTrainer(model="WeiboAI/VibeThinker-1.5B", train_dataset=qwen_coder_trajectories_subdataset)
=======
from data_preprocessing import dataset

trainer = SFTTrainer(model="WeiboAI/VibeThinker-1.5B", train_dataset=dataset)
>>>>>>> aae5206e73bfc7b071513773543380bc0c95f7e3
trainer.train()