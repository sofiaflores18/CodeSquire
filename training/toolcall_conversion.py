"""Convert mini-coder-trajs-400k trajectories into mini-swe-agent's toolcall format.

The SFT data was recorded against mini-swe-agent's v1-style *text* protocol: one
```bash fenced block per assistant turn, observations returned as `user` messages.
mini-swe-agent 2.4.6 (what run_swebench.sh / run_eval.py actually run) uses
toolcalls instead -- `minisweagent/models/utils/actions_text.py` calls the text
path legacy and says "As of mini-swe-agent v2.0, we strongly recommend to use
toolcalls instead". Training on the old protocol and evaluating on the new one is
a train/serve mismatch, so convert the data to what eval sends:

    assistant  content = reasoning, tool_calls = [bash({"command": ...})]
    tool       content = the observation, tool_call_id = the matching call

The prompts are read from the installed mini-swe-agent rather than vendored, so
this cannot silently drift from the agent that produces the eval trajectories.
"""

import json
import re
from pathlib import Path

import yaml

# minisweagent/models/utils/actions_toolcall.py
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"}
            },
            "required": ["command"],
        },
    },
}

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
# The task text the recorded instance_template wrapped; we re-wrap it in the current one.
TASK_RE = re.compile(
    r"<pr_description>\s*Consider the following PR description:\s*(.*?)\s*</pr_description>",
    re.DOTALL,
)
# ```bash ... ``` (the recorded protocol's action)
FENCE_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n?```", re.DOTALL)


def find_mini_config(explicit: str | None = None) -> Path:
    """Locate mini-swe-agent's swebench.yaml, so prompts match the eval harness."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--mini-config not found: {path}")
        return path
    try:  # installed in this interpreter
        import minisweagent

        candidate = Path(minisweagent.__file__).parent / "config/benchmarks/swebench.yaml"
        if candidate.is_file():
            return candidate
    except ImportError:
        pass
    # Installed as a uv tool (its own venv), which is how the README sets it up.
    matches = sorted(
        Path.home().glob(
            ".local/share/uv/tools/mini-swe-agent/lib/python*/site-packages/"
            "minisweagent/config/benchmarks/swebench.yaml"
        )
    )
    if matches:
        return matches[-1]
    raise FileNotFoundError(
        "Could not find mini-swe-agent's swebench.yaml. Install it "
        "(`uv tool install 'mini-swe-agent==2.4.6'`) or pass an explicit path. "
        "The prompts are deliberately not vendored: a stale copy would silently "
        "reintroduce the train/serve mismatch this conversion exists to fix."
    )


def load_templates(config_path: Path) -> dict:
    agent = yaml.safe_load(config_path.read_text())["agent"]
    return {
        "system": agent["system_template"],
        "instance": agent["instance_template"],
    }


def _render_instance(template: str, task: str) -> str:
    """Render instance_template. Only `task` is substituted; nothing else in the
    swebench template uses Jinja, so a literal replace avoids importing jinja2
    and avoids StrictUndefined errors on the agent's other template vars."""
    return template.replace("{{task}}", task)


def _split_turn(content: str) -> tuple[str, str] | None:
    """Split a recorded assistant turn into (reasoning, command)."""
    match = FENCE_RE.search(content)
    if not match:
        return None
    command = match.group(1).strip()
    if not command:
        return None
    reasoning = content[: match.start()].strip()
    return reasoning, command


def _assistant(reasoning: str, command: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": reasoning,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }


def _tool(content: str, call_id: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _observation(output: str, returncode: int = 0) -> str:
    """Mirror swebench.yaml's observation_template for outputs under its 10k cutoff."""
    return f"<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"


def _patched_files(patch: str) -> list[str]:
    """Source paths from a unified diff, for the `git diff -- <files>` submit step."""
    return re.findall(r"^diff --git a/(\S+) b/\S+", patch or "", re.MULTILINE)


def _submit_turns(patch: str, next_id) -> list[dict]:
    """The current protocol's submission: create patch.txt, verify it, then submit.

    The recorded trajectories submitted in a single `echo MARKER && git add -A &&
    git diff --cached`, which the current instance_template explicitly forbids
    (the steps must be separate commands). We rebuild the three steps; the patch
    body is the trajectory's real patch, so only the plumbing is synthesized.
    """
    files = _patched_files(patch)
    if not files:
        return []
    diff_cmd = "git diff -- " + " ".join(files) + " > patch.txt"
    messages = []

    call = next_id()
    messages.append(_assistant(
        "The fix is in place. Let me create the patch file containing only the "
        "source files I modified.", diff_cmd, call))
    messages.append(_tool(_observation(""), call))

    call = next_id()
    messages.append(_assistant(
        "Now let me verify the patch contains only my intended changes.",
        "cat patch.txt", call))
    messages.append(_tool(_observation(patch.strip()), call))

    call = next_id()
    messages.append(_assistant(
        "The patch looks correct and contains only the source changes needed to "
        "fix the issue. Submitting.",
        f"echo {SUBMIT_MARKER} && cat patch.txt", call))
    return messages


def to_toolcall_messages(messages: list[dict], patch: str, templates: dict) -> list[dict] | None:
    """Convert one recorded trajectory. Returns None if it cannot be converted cleanly."""
    if not messages or messages[0]["role"] != "system":
        return None
    task_match = TASK_RE.search(messages[1]["content"] if len(messages) > 1 else "")
    if not task_match:
        return None

    counter = iter(range(10_000))
    next_id = lambda: f"call_{next(counter):04d}"  # noqa: E731 - deterministic, replay-stable

    out = [
        {"role": "system", "content": templates["system"].rstrip()},
        {"role": "user", "content": _render_instance(templates["instance"], task_match.group(1))},
    ]

    # Walk the recorded assistant/observation pairs, stopping at the old submit turn.
    index = 2
    while index < len(messages):
        message = messages[index]
        if message["role"] != "assistant":
            return None  # unexpected shape; drop rather than guess
        if SUBMIT_MARKER in message["content"]:
            break
        split = _split_turn(message["content"])
        if split is None:
            return None
        call = next_id()
        out.append(_assistant(split[0], split[1], call))
        index += 1
        # Its observation, passed through verbatim: the recorded bodies already
        # match swebench.yaml's observation_template.
        if index < len(messages) and messages[index]["role"] == "user":
            out.append(_tool(messages[index]["content"], call))
            index += 1
    else:
        return None  # never reached the submit turn

    submit = _submit_turns(patch, next_id)
    if not submit:
        return None
    out.extend(submit)
    return out
