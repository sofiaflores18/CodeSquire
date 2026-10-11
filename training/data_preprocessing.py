'''
Preprocessing the data from https://huggingface.co/datasets/ricdomolm/mini-coder-trajs-400k/viewer/default/train?row=0

Qwen-3-Coder-30B trajectories
'''

from datasets import load_dataset

dataset = load_dataset("ricdomolm/mini-coder-trajs-400k", split="train")

# Keep successful trajectories
dataset = dataset.filter(lambda x: x["verified"] is True)

# Keep only the conversational data needed for SFT
dataset = dataset.select_columns(["messages"])

print(dataset)
print(dataset[0]["messages"][:3])