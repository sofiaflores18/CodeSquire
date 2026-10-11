'''
Preprocessing the data from https://huggingface.co/datasets/ricdomolm/mini-coder-trajs-400k/viewer/default/train?row=0

Qwen-3-Coder-30B trajectories
'''

from datasets import load_dataset

def data_preprocessing_SFT(num_samples: int):
    dataset = load_dataset("ricdomolm/mini-coder-trajs-400k", split="train")

    # Keep successful trajectories
    dataset = dataset.filter(lambda x: x["verified"] is True)

    # Keep only the conversational data needed for SFT
    dataset = dataset.select_columns(["messages"])

    dataset = dataset.shuffle(seed=11)

    qwen_coder_trajectories_subdataset = dataset.select(num_samples)
    print(qwen_coder_trajectories_subdataset)
    print(qwen_coder_trajectories_subdataset[0]["messages"][:3])
