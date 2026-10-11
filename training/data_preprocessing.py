'''
Preprocessing the data from https://huggingface.co/datasets/ricdomolm/mini-coder-trajs-400k/viewer/default/train?row=0

Qwen-3-Coder-30B trajectories
'''

import math

from datasets import load_dataset
from transformers import AutoTokenizer

# Trajectories are long: median ~9.1k tokens, p90 ~16.1k, max ~45k (Qwen3 tokenizer).
# 16384 keeps 90% of them whole and 95% of all tokens; 32768 only buys 5% more
# tokens for double the activation memory.
DEFAULT_MAX_LENGTH = 16384

# Fraction of the selected trajectories held out to detect memorization.
DEFAULT_EVAL_FRACTION = 0.05

# The length filter rejects ~10% of trajectories, so over-sample the candidate
# pool before filtering to still end up with num_samples.
CANDIDATE_MULTIPLIER = 1.5

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _num_tokens(tokenizer, messages) -> int:
    """Length of a trajectory once the chat template is applied."""
    encoded = tokenizer.apply_chat_template(messages, tokenize=True)
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
    return (
        len(messages) >= 2
        and messages[0]["role"] == "system"
        and messages[-1]["role"] == "assistant"
        and SUBMIT_MARKER in messages[-1]["content"]
    )


def data_preprocessing_SFT(
    num_samples: int,
    model_name: str = "Qwen/Qwen3-8B",
    max_length: int = DEFAULT_MAX_LENGTH,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    seed: int = 11,
):
    """Return a DatasetDict with 'train' and 'test' splits of complete trajectories.

    Only trajectories that fit in max_length are kept, so every example the model
    sees ends in a successful submit. The system and user turns stay in the
    sequence as context -- SFTConfig(assistant_only_loss=True) masks them out of
    the loss rather than removing them, since the assistant turns need them to
    condition on.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("ricdomolm/mini-coder-trajs-400k", split="train")

    # Keep successful trajectories
    dataset = dataset.filter(lambda x: x["verified"] is True)

    # Keep only the conversational data needed for SFT
    dataset = dataset.select_columns(["messages"])

    # Arbitrary seed
    dataset = dataset.shuffle(seed=seed)

    # Tokenizing all ~180k verified trajectories to measure length would take
    # far longer than the training run, so only measure a candidate pool and
    # grow it if too few survive the filter.
    candidates = min(len(dataset), math.ceil(num_samples * CANDIDATE_MULTIPLIER))
    while True:
        pool = dataset.select(range(candidates))
        kept = pool.filter(
            lambda x: _is_complete(x["messages"])
            and _num_tokens(tokenizer, x["messages"]) <= max_length
        )
        if len(kept) >= num_samples or candidates == len(dataset):
            break
        # Scale the pool by the observed survival rate, with headroom.
        candidates = min(len(dataset), math.ceil(candidates * num_samples / max(len(kept), 1) * 1.1))

    if len(kept) < num_samples:
        raise ValueError(
            f"Only {len(kept)} trajectories of <= {max_length} tokens in the whole "
            f"dataset, but {num_samples} were requested. Raise max_length or lower "
            f"--num_samples."
        )

    print(
        f"Kept {len(kept)}/{len(pool)} candidates at max_length={max_length} "
        f"({len(kept) / len(pool):.1%} survival); selecting {num_samples}."
    )
    qwen_coder_trajectories_subdataset = kept.select(range(num_samples))

    splits = qwen_coder_trajectories_subdataset.train_test_split(test_size=eval_fraction, seed=seed)
    print(splits)
    print(splits["train"][0]["messages"][:3])

    return splits
