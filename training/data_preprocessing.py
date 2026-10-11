'''
Preprocessing the data from https://huggingface.co/datasets/ricdomolm/mini-coder-trajs-400k/viewer/default/train?row=0

Qwen-3-Coder-30B trajectories

The recorded trajectories use mini-swe-agent's legacy text protocol (one ```bash
block per turn, observations as `user` messages). The eval harness runs
mini-swe-agent 2.4.6, which uses toolcalls. toolcall_conversion rewrites the
trajectories into the toolcall format so training and eval agree; see that
module for the details.
'''

import math

from datasets import load_dataset
from transformers import AutoTokenizer

from toolcall_conversion import (
    BASH_TOOL,
    SUBMIT_MARKER,
    find_mini_config,
    load_templates,
    to_toolcall_messages,
)

# Trajectories are long: in toolcall form the median is ~9k tokens and the tail
# runs past 40k. 16384 keeps ~90% of them whole and ~95% of all tokens; 32768
# only buys a few percent more for double the activation memory.
DEFAULT_MAX_LENGTH = 16384

# Fraction of the selected trajectories held out to detect memorization.
DEFAULT_EVAL_FRACTION = 0.05

# Conversion and the length filter both reject some trajectories, so over-sample
# the candidate pool to still end up with num_samples.
CANDIDATE_MULTIPLIER = 1.5


def _num_tokens(tokenizer, messages, tools) -> int:
    """Length of a trajectory once the chat template and tool schema are applied."""
    encoded = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True)
    # transformers>=5 hands back a BatchEncoding; older versions a flat list of ids.
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def _is_complete(messages) -> bool:
    """Keep only trajectories that run start to finish.

    Truncating a trajectory would drop its final submit turn and teach the model
    to never finish -- the exact failure we are trying to train away. So we drop
    over-long trajectories instead of letting the trainer cut them short.
    """
    if len(messages) < 2 or messages[0]["role"] != "system":
        return False
    last = messages[-1]
    if last["role"] != "assistant" or not last.get("tool_calls"):
        return False
    return SUBMIT_MARKER in last["tool_calls"][0]["function"]["arguments"]


def data_preprocessing_SFT(
    num_samples: int,
    model_name: str = "Qwen/Qwen3-8B",
    max_length: int = DEFAULT_MAX_LENGTH,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    seed: int = 11,
    mini_config: str | None = None,
):
    """Return a DatasetDict with 'train' and 'test' splits of complete trajectories.

    Rows carry `messages` (toolcall format) and `tools` (the bash schema), which is
    what the chat template needs to render tool calls. The system and user turns
    stay in the sequence as context -- SFTConfig(assistant_only_loss=True) masks
    them out of the loss rather than removing them, since the assistant turns need
    them to condition on.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config_path = find_mini_config(mini_config)
    print(f"Agent prompts from {config_path}")
    templates = load_templates(config_path)

    dataset = load_dataset("ricdomolm/mini-coder-trajs-400k", split="train")

    # Keep successful trajectories
    dataset = dataset.filter(lambda x: x["verified"] is True)

    # `patch` is needed to rebuild the submit steps; dropped after conversion.
    dataset = dataset.select_columns(["messages", "patch"])

    # Arbitrary seed
    dataset = dataset.shuffle(seed=seed)

    def convert(example):
        converted = to_toolcall_messages(example["messages"], example["patch"], templates)
        # Mark unconvertible rows with an empty list so a later .filter drops them.
        return {"messages": converted or [], "tools": [BASH_TOOL]}

    # Converting and tokenizing all ~180k verified trajectories would take far
    # longer than the training run, so only process a candidate pool and grow it
    # if too few survive.
    candidates = min(len(dataset), math.ceil(num_samples * CANDIDATE_MULTIPLIER))
    while True:
        pool = dataset.select(range(candidates)).map(convert, remove_columns=["patch"])
        kept = pool.filter(
            lambda x: _is_complete(x["messages"])
            and _num_tokens(tokenizer, x["messages"], x["tools"]) <= max_length
        )
        if len(kept) >= num_samples or candidates == len(dataset):
            break
        # Scale the pool by the observed survival rate, with headroom.
        candidates = min(len(dataset), math.ceil(candidates * num_samples / max(len(kept), 1) * 1.1))

    if len(kept) < num_samples:
        raise ValueError(
            f"Only {len(kept)} convertible trajectories of <= {max_length} tokens, but "
            f"{num_samples} were requested. Raise max_length or lower --num_samples."
        )

    print(
        f"Kept {len(kept)}/{len(pool)} candidates at max_length={max_length} "
        f"({len(kept) / len(pool):.1%} survival); selecting {num_samples}."
    )
    qwen_coder_trajectories_subdataset = kept.select(range(num_samples))

    splits = qwen_coder_trajectories_subdataset.train_test_split(
        test_size=eval_fraction, seed=seed
    )
    print(splits)
    return splits
