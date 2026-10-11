#!/usr/bin/env python3
"""CodeSquire: prompt the CodeSquire agent from your terminal and let it write code.

The agent runs on a vLLM server (OpenAI-compatible API). This CLI is the other half
of the loop that `run_eval.py` runs on SWE-bench: it reads the model's command,
executes it in your shell, and feeds the result back until the task is done.

    codesquire                            # interactive session
    codesquire "add type hints to utils.py"
    codesquire --api-base http://VM:8000/v1 "fix the failing test"

Stdlib only, single file: copy it to any machine that can reach the server.

The prompt format is the one the model was fine-tuned on (mini-coder trajectories):
a THOUGHT section, then exactly one ```bash block, then a <returncode>/<output>
observation. Changing those templates will take the model off-distribution.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

VERSION = "0.1.0"

DEFAULT_API_KEY = os.environ.get("CODESQUIRE_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))

# Where to look when nobody says. 8080 is the port the VM tunnel forwards; 8000 is vLLM's
# own default and what run_eval.py uses locally. Trying both means the demo works either
# way instead of failing with a connection error that looks like the server is down.
DEFAULT_PORTS = (8080, 8000)


def default_endpoints() -> list[str]:
    """Endpoints to try, in order, when no --api-base was given."""
    configured = os.environ.get("CODESQUIRE_API_BASE")
    if configured:
        return [configured]
    return [f"http://localhost:{port}/v1" for port in DEFAULT_PORTS]

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# vLLM truncates at max_model_len; keep observations well under it (training used 10000).
OBSERVATION_CHAR_LIMIT = 10000
# What we show the user per command; the model still sees OBSERVATION_CHAR_LIMIT.
DISPLAY_CHAR_LIMIT = 2000


# --------------------------------------------------------------------------- output

class Style:
    """ANSI codes, disabled when stdout is not a terminal or NO_COLOR is set."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        return "".join(f"\033[{c}m" for c in codes) + text + "\033[0m"

    def prompt(self, text: str, *codes: str) -> str:
        """Like __call__, but marks the escapes invisible so readline counts columns right."""
        if not self.enabled or not codes:
            return text
        return "".join(f"\001\033[{c}m\002" for c in codes) + text + "\001\033[0m\002"

    def dim(self, t): return self(t, "2")
    def bold(self, t): return self(t, "1")
    def cyan(self, t): return self(t, "36")
    def green(self, t): return self(t, "32")
    def yellow(self, t): return self(t, "33")
    def red(self, t): return self(t, "31")
    def magenta(self, t): return self(t, "35")


S = Style(sys.stdout.isatty() and not os.environ.get("NO_COLOR"))


# Six rows, 78 columns, every row the same width. Shown only when the terminal is wide
# enough; a narrower one gets the name on a single line instead of a wrapped mess.
WORDMARK = (
    " ██████╗ ██████╗ ██████╗ ███████╗███████╗ ██████╗ ██╗   ██╗██╗██████╗ ███████╗",
    "██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔═══██╗██║   ██║██║██╔══██╗██╔════╝",
    "██║     ██║   ██║██║  ██║█████╗  ███████╗██║   ██║██║   ██║██║██████╔╝█████╗  ",
    "██║     ██║   ██║██║  ██║██╔══╝  ╚════██║██║▄▄ ██║██║   ██║██║██╔══██╗██╔══╝  ",
    "╚██████╗╚██████╔╝██████╔╝███████╗███████║╚██████╔╝╚██████╔╝██║██║  ██║███████╗",
    " ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚═╝  ╚═╝╚══════╝",
)
WORDMARK_WIDTH = len(WORDMARK[0])


def gradient() -> tuple[str, ...]:
    """One colour per wordmark row, blue to magenta.

    256-colour codes look best but are not universal, so a terminal that does not
    advertise them gets the basic-colour approximation instead of possible garbage.
    """
    if not S.enabled:
        return ()
    rich = "256color" in os.environ.get("TERM", "") or bool(os.environ.get("COLORTERM"))
    if rich:
        return ("38;5;39", "38;5;75", "38;5;111", "38;5;141", "38;5;177", "38;5;213")
    return ("96", "36", "36", "35", "35", "95")


def short_path(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path == home or path.startswith(home + os.sep) else path


def print_banner(cfg: Config, sandbox: "Sandbox | None") -> None:
    """The splash an interactive session opens with."""
    width = shutil.get_terminal_size((80, 24)).columns
    colours = gradient()

    if width >= WORDMARK_WIDTH + 2:
        indent = "  " if width >= WORDMARK_WIDTH + 4 else ""
        print()
        for row, code in zip(WORDMARK, colours or ("",) * len(WORDMARK)):
            print(indent + (S(row, code) if code else row))
    else:
        name = "CodeSquire"
        if colours:
            # Bold has to go on each character: one reset inside the string would
            # otherwise cancel a single bold wrapped around the whole thing.
            print("\n  " + "".join(S(char, "1", colours[index * len(colours) // len(name)])
                                   for index, char in enumerate(name)))
        else:
            print("\n  " + S.bold(name))

    # Wide enough for the longest label ("approval") plus a gap before the value.
    column = 9
    # One column of slack: a line that exactly fills the terminal leaves the cursor in
    # a wrapped state, which some terminals turn into a stray blank line.
    room = max(width - 3 - column, 20)

    def fit(value: str, keep_tail: bool = False) -> str:
        """Trim a value to the terminal, keeping the end of a path and the start of a name."""
        if len(value) <= room:
            return value
        return "\u2026" + value[-(room - 1):] if keep_tail else value[:room - 1] + "\u2026"

    context = f"  (context {cfg.max_context_tokens:,})"
    rows = [
        ("model", fit(cfg.model, keep_tail=True)),
        # The annotation is the first thing to go when the line will not fit.
        ("server", cfg.api_base + context if len(cfg.api_base + context) <= room
                   else fit(cfg.api_base)),
        ("cwd", fit(short_path(cfg.cwd), keep_tail=True)),
    ]
    if cfg.docker_image:
        detail = "shares the cwd only" if cfg.network else "shares the cwd only, no network"
        full = f"{cfg.docker_image}  ({detail})"
        rows.append(("sandbox", full if len(full) <= room else fit(cfg.docker_image)))
    print(S.dim(f"  v{VERSION}"))
    for label, value in rows:
        print(S.dim(f"  {label:<{column}}") + value)
    if cfg.docker_image and sandbox is not None and sandbox.missing:
        print(S.dim(f"  {'missing':<{column}}")
              + S.dim(fit(", ".join(sandbox.missing) + " (not in this image)")))
    if cfg.auto_approve:
        print(S.dim(f"  {'approval':<{column}}")
              + S.yellow(fit("off — commands run without asking")))
    print()
    print(S.dim("  /help for commands, /exit to quit"))


def rule(label: str = "") -> str:
    width = min(shutil.get_terminal_size((80, 24)).columns, 100)
    if not label:
        return S.dim("─" * width)
    return S.dim("── " + label + " " + "─" * max(width - len(label) - 4, 0))


# --------------------------------------------------------------------------- prompts

# Verbatim from the training trajectories. The model is conditioned on this wording.
SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <example>.

<example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</example>

Failure to follow these rules will cause your response to be rejected."""

# The task framing is general purpose: the model was fine-tuned on SWE-bench-style
# trajectories, but it is a shell agent and the harness here is a plain terminal, so the
# PR-description and git-patch scaffolding is gone. What is kept verbatim is the part the
# fine-tune actually conditioned on: THOUGHT, exactly one bash block, and the submit marker.
INSTANCE_TEMPLATE = """\
<task>
{task}
</task>

<instructions>
# Task Instructions

## Overview

You're a capable engineer with a shell on the user's computer. Help with whatever the
task above asks: writing or changing code, reading and explaining a codebase, running
commands, debugging, answering a question, or anything else.

<IMPORTANT>This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.</IMPORTANT>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide exactly ONE bash command to execute

## Recommended Workflow

1. Orient yourself: look at the directory and read whatever files are relevant
2. Do the work in small steps, checking the result of each one
3. If you wrote or changed code, run it to confirm it works
4. Tell the user what you did, then finish

Keep it proportional: a one-line question deserves one command, not an investigation.

## Command Execution Rules

You are operating in an environment where

1. You write a single command
2. The system executes that command in a subshell
3. You see the result
4. You write your next command

Each response should include:

1. A **THOUGHT** section where you explain your reasoning and plan
2. A single bash code block with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL
- Do NOT try to run multiple independent commands in separate blocks in one response
- NEVER use an interactive editor (vi, vim, nano, emacs, less, more, top). There is no
  terminal to drive them. Write files with a `cat <<'SCRIPT_EOF' > file` heredoc and
  change them with `sed -i`, as shown below
- Do NOT repeat a command that already failed. Read the error and try something different
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

```bash
ls -la
```
</example>

If you need to run multiple commands, either:
1. Combine them in one block using && or ||
```bash
command1 && command2 || echo "Error occurred"
```

2. Wait for the first command to complete, see its output, then issue the next command in your following response.

## Environment Details
- Working directory for every command: {cwd}
- Platform: {platform}{sandbox}
- A command's output is truncated if it is very long, so prefer selective output
- Always use non-interactive flags (-y, -f) for commands
- The user sees every command before it runs and may decline it

## Useful Command Examples

### Create a new file:
```bash
cat <<'SCRIPT_EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
SCRIPT_EOF
```

### Edit files with sed:
# Replace all occurrences
```bash
sed -i 's/old_string/new_string/g' filename.py
```

# Replace only first occurrence
```bash
sed -i 's/old_string/new_string/' filename.py
```

# Replace all occurrences in lines 1-10
```bash
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:
```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run
```bash
anything
```

## Finishing
You have two ways to end your turn, and you must use one of them:

1. If you ran commands, write a final message that answers the task or summarizes what
   you changed, and issue exactly this command, on its own:

```bash
echo {marker}
```

2. If the task needs no commands at all -- the user asked a question you can already
   answer -- reply with the answer and NO bash code block. A response with no command
   is treated as your final answer.

Either way your turn ends there, so say what the user needs to know before you finish.
</instructions>"""

FOLLOW_UP_TEMPLATE = """\
<task>
{task}
</task>

Continue in the same working directory, in the same format: a THOUGHT section and
exactly one bash code block, or a direct answer with no command if none is needed."""

SANDBOX_NOTE = """
- Commands run inside a disposable Docker container ({image}); only {cwd} is shared with
  the user's computer, so work there and expect anything written elsewhere to vanish
- Tools present: {present}
- Not installed: {missing} -- do not reach for these, and do not try to apt-get them
- `pip install --user <package>` works if you need a Python library"""

FORMAT_ERROR_TEMPLATE = """\
Format error:

<error>
{error}
</error>

Please always provide EXACTLY ONE action in triple backticks, found {count} actions.

Please format your action in triple backticks as shown in <example>.

<example>
THOUGHT: Here are some thoughts about why you want to perform the action.

```bash
<action>
```
</example>

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that)."""

# A small model that gets nothing back from a command will often run it forever.
# scripts/swebench_runner.py ends the task silently after 8 repeats, which is right for
# scoring a benchmark; here somebody is waiting for an answer, so nudge first and only
# give up if the nudge does not land.
NUDGE_AFTER = 3
STOP_AFTER = 8
# Consecutive malformed responses to ask about before working with what was sent.
FORMAT_RETRIES = 2

REPEAT_NUDGE = (
    "\n<warning>\nYou have now run this exact command {count} times and the result is not "
    "changing. Stop repeating it. Either try a genuinely different command, or, if you "
    "cannot make further progress, say so and finish.\n</warning>"
)

LENGTH_ERROR = (
    "Your previous response reached the output token limit before you produced a "
    "complete action, so it was cut off. Respond more concisely and provide exactly "
    "one action in the required format. If you need to think more, do so briefly."
)


def format_observation(returncode: int, output: str) -> str:
    """The <returncode>/<output> observation the model was trained to read."""
    if len(output) < OBSERVATION_CHAR_LIMIT:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
    half = OBSERVATION_CHAR_LIMIT // 2
    elided = len(output) - OBSERVATION_CHAR_LIMIT
    return (
        f"<returncode>{returncode}</returncode>\n"
        "<warning>\n"
        "The output of your last command was too long.\n"
        "Please try a different command that produces less output.\n"
        "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
        "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
        "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.\n"
        "</warning>\n"
        f"<output_head>\n{output[:half]}\n</output_head>\n"
        f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-half:]}\n</output_tail>"
    )


# --------------------------------------------------------------------------- config

@dataclass
class Config:
    api_base: str = ""                  # resolved from --api-base, the env var, or a probe
    api_key: str = DEFAULT_API_KEY
    model: str = ""
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 2048
    thinking: bool | None = None        # None: don't send chat_template_kwargs at all
    step_limit: int = 40
    command_timeout: int = 120
    cwd: str = field(default_factory=os.getcwd)
    auto_approve: bool = False
    stream: bool = True
    docker_image: str = ""              # empty: run commands directly on this machine
    network: bool = True
    request_timeout: int = 600
    max_context_tokens: int = 32768     # refined from /v1/models when available


# --------------------------------------------------------------------------- server

class ServerError(RuntimeError):
    pass


def _request(url: str, cfg: Config, payload: dict | None = None, timeout: int | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key or 'EMPTY'}",
        },
        method="POST" if data else "GET",
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout or cfg.request_timeout)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")[:2000]
        raise ServerError(f"HTTP {error.code} from {url}\n{body}") from None
    except urllib.error.URLError as error:
        raise ServerError(f"Cannot reach the model server at {cfg.api_base} ({error.reason}).") from None
    except TimeoutError:
        raise ServerError(f"Request to {url} timed out after {timeout or cfg.request_timeout}s") from None


UNREACHABLE_HELP = """\
Open the tunnel to the VM and start vLLM inside that session
(the exact commands for this project are in cli/README.md):
  ssh -p <ssh-port> root@<vm-host> -L 8080:localhost:8080
  vllm serve <model> --port 8080 --max-model-len 16384
Already serving somewhere else? Name it:
  codesquire --api-base http://<host>:<port>/v1"""


def resolve_endpoint(cfg: Config, candidates: list[str]) -> tuple[Config, str, int | None]:
    """Settle on the endpoint that is actually serving, and what it serves.

    With one candidate its error is the whole story. With several, all of them failing
    is the interesting case, so the report names each one rather than only the last.
    """
    failures = []
    for base in candidates:
        probe = replace(cfg, api_base=base)
        try:
            model_id, length = discover_model(probe)
            return probe, model_id, length
        except ServerError as error:
            failures.append((base, error))

    if len(failures) == 1:
        raise ServerError(f"{failures[0][1]}\n\n{UNREACHABLE_HELP}")
    tried = "\n".join(f"  {base}" for base, _ in failures)
    raise ServerError(f"No model server responded. Tried:\n{tried}\n\n{UNREACHABLE_HELP}")


def discover_model(cfg: Config) -> tuple[str, int | None]:
    """Return (model id, max_model_len) from /v1/models so the user need not type the HF id."""
    with _request(f"{cfg.api_base.rstrip('/')}/models", cfg, timeout=30) as response:
        body = json.load(response)
    served = body.get("data") or []
    if not served:
        raise ServerError(f"The server at {cfg.api_base} reports no served models.")
    if cfg.model:
        ids = [m.get("id") for m in served]
        if cfg.model not in ids:
            raise ServerError(f"Model {cfg.model!r} is not served. Available: {', '.join(map(str, ids))}")
        chosen = next(m for m in served if m.get("id") == cfg.model)
    else:
        chosen = served[0]
    length = chosen.get("max_model_len")
    return chosen["id"], length if isinstance(length, int) else None


def complete(cfg: Config, messages: list[dict], on_text=None) -> tuple[str, str]:
    """One chat completion. Returns (text, finish_reason).

    on_text is called with each new chunk while streaming so the terminal can show
    the model thinking instead of sitting silent for 30 seconds.
    """
    extra_body = {}
    if cfg.top_k is not None and cfg.top_k > 0:
        extra_body["top_k"] = cfg.top_k          # vLLM-specific, not in the OpenAI schema
    if cfg.min_p:
        extra_body["min_p"] = cfg.min_p
    if cfg.thinking is not None:
        # Qwen3 thinks by default; its chat template reads this kwarg.
        extra_body["chat_template_kwargs"] = {"enable_thinking": cfg.thinking}

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
        "stream": cfg.stream,
        **extra_body,
    }
    url = f"{cfg.api_base.rstrip('/')}/chat/completions"

    if not cfg.stream:
        with _request(url, cfg, payload) as response:
            body = json.load(response)
        choice = body["choices"][0]
        text = choice["message"].get("content") or ""
        if on_text:
            on_text(text)
        return text, choice.get("finish_reason") or "stop"

    parts: list[str] = []
    finish_reason = "stop"
    with _request(url, cfg, payload) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                # Some servers stream reasoning separately; it is not part of the action.
                piece = delta.get("content") or ""
                if piece:
                    parts.append(piece)
                    if on_text:
                        on_text(piece)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    return "".join(parts), finish_reason


# --------------------------------------------------------------------------- parsing

# The trained fence is ```bash; accept the other mini-swe-agent fences and a bare fence
# so a small model's formatting slip does not cost a step.
ACTION_RE = re.compile(
    r"```(?:bash|mswea_bash_command|sh|shell|console)?[ \t]*\r?\n(.*?)(?:\r?\n)?```",
    re.DOTALL,
)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Models imitate the <example> wrappers in the prompt and emit them around their real
# answer. Only the tags are dropped, never what is inside them: the real command often
# ends up in there.
SCAFFOLD_TAG_RE = re.compile(
    r"[ \t]*</?(?:format_)?example(?:_response)?>[ \t]*\n?", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Drop <think> blocks and echoed prompt scaffolding: neither is the action."""
    text = THINK_RE.sub("", text)
    # An unterminated <think> means the response was cut off mid-reasoning.
    if "<think>" in text:
        text = text.split("<think>")[0]
    return SCAFFOLD_TAG_RE.sub("", text)


def parse_actions(text: str) -> list[str]:
    return [m.group(1).strip() for m in ACTION_RE.finditer(strip_reasoning(text)) if m.group(1).strip()]


def split_thought(text: str) -> str:
    """The prose before the first code fence, with the THOUGHT: label removed."""
    thought = strip_reasoning(text).split("```")[0].strip()
    if thought.upper().startswith("THOUGHT:"):
        thought = thought[len("THOUGHT:"):].strip()
    return thought


def is_submit(command: str) -> bool:
    return SUBMIT_MARKER in command


# --------------------------------------------------------------------------- execution

# Commands whose blast radius reaches outside the working tree, or that are
# irreversible. These always ask, even with --yes.
# (pattern, why, contained) -- `contained` marks the ones a Docker sandbox already
# neutralises, so they stop being worth a prompt once --docker is on.
DANGEROUS_PATTERNS = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*(/|~|\$HOME)(\s|$)", "deletes a path outside the project", True),
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f|\brm\s+-[a-zA-Z]*f[a-zA-Z]*r", "recursive force delete", False),
    (r"\b(mkfs|fdisk|parted)\b", "formats a disk", True),
    (r"\bdd\s+.*\bof=/dev/", "writes directly to a device", True),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "powers off the machine", True),
    (r":\(\)\s*\{.*\};:", "fork bomb", True),
    (r"\bchmod\s+(-[a-zA-Z]*\s+)*777\s+/", "opens permissions at the filesystem root", True),
    (r"\bcurl\b[^|;&]*\|\s*(sudo\s+)?(ba)?sh", "pipes a download into a shell", False),
    (r"\bwget\b[^|;&]*\|\s*(sudo\s+)?(ba)?sh", "pipes a download into a shell", False),
    (r"\bsudo\b", "runs as root", True),
    (r"\bgit\s+(push|reset\s+--hard|clean\s+-[a-zA-Z]*f)", "rewrites git state or the remote", False),
    (r">\s*/dev/(sd|nvme|disk)", "writes directly to a device", True),
    (r"\b(shred|srm)\b", "irrecoverable delete", False),
]


def risk_of(command: str, sandboxed: bool = False) -> str | None:
    """Why this command needs a human, or None. A sandbox excuses the contained ones.

    What a container does not contain: the bind-mounted working directory, and the
    network -- so deletes, force-pushes and piped downloads still ask.
    """
    for pattern, why, contained in DANGEROUS_PATTERNS:
        if sandboxed and contained:
            continue
        if re.search(pattern, command):
            return why
    return None


class Sandbox:
    """Runs the agent's commands inside a disposable container.

    The working directory is bind-mounted at the same absolute path it has on the host,
    so paths the model reads and writes mean the same thing on both sides and nothing
    outside that directory is reachable. One container is kept alive for the session:
    installing a package in one step is still there in the next, which a fresh container
    per command would throw away.
    """

    # Worth telling the model about up front: a slim image has no git, and a model that
    # does not know that burns steps discovering it.
    PROBED_TOOLS = ("python3", "pip", "git", "curl", "wget", "gcc", "make", "node", "jq")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.name = f"codesquire-{os.getpid()}"
        self.started = False
        self.present: list[str] = []
        self.missing: list[str] = []

    def preflight(self) -> None:
        if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
            raise RuntimeError(
                "Docker is not usable (is the daemon running, and is your user in the "
                "docker group?). Drop --docker to run commands directly instead."
            )
        present = subprocess.run(
            ["docker", "image", "inspect", self.cfg.docker_image],
            capture_output=True,
        )
        if present.returncode != 0:
            print(S.dim(f"  pulling {self.cfg.docker_image} (first run only)…"), flush=True)
            if subprocess.run(["docker", "pull", self.cfg.docker_image]).returncode != 0:
                raise RuntimeError(f"Could not pull {self.cfg.docker_image}")

    def start(self) -> None:
        self.preflight()
        command = [
            "docker", "run", "--rm", "--detach", "--name", self.name,
            # Same path inside and out, so the model never has to translate paths.
            "--volume", f"{self.cfg.cwd}:{self.cfg.cwd}",
            "--workdir", self.cfg.cwd,
            # Files the agent creates stay owned by the user, not root.
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--env", "HOME=/tmp",
            "--env", "PYTHONUSERBASE=/tmp/pyuser",
            "--env", f"PATH=/tmp/pyuser/bin:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
            "--pids-limit", "1024",          # a runaway loop cannot fork-bomb the host
        ]
        if not self.cfg.network:
            command += ["--network", "none"]
        command += ["--entrypoint", "sleep", self.cfg.docker_image, "infinity"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Could not start the sandbox container:\n{result.stderr.strip()}")
        self.started = True
        self.probe()

    def probe(self) -> None:
        """Ask the image which of the usual tools it actually has."""
        script = "; ".join(f"command -v {tool} >/dev/null && echo {tool}"
                           for tool in self.PROBED_TOOLS)
        found = self.run(script)[1].split()
        self.present = [tool for tool in self.PROBED_TOOLS if tool in found]
        self.missing = [tool for tool in self.PROBED_TOOLS if tool not in found]

    def note(self) -> str:
        return SANDBOX_NOTE.format(
            image=self.cfg.docker_image,
            cwd=self.cfg.cwd,
            present=", ".join(self.present) or "a minimal shell only",
            missing=", ".join(self.missing) or "nothing of note",
        )

    def run(self, command: str) -> tuple[int, str]:
        # The command goes in on stdin as a script, so no amount of quoting or heredocs
        # in it can break out of the argv we build here. `timeout` runs inside the
        # container, so a hung command is killed with its children rather than leaving
        # an orphan behind a dead `docker exec`.
        result = subprocess.run(
            ["docker", "exec", "--interactive", self.name,
             "timeout", "--kill-after", "5", str(self.cfg.command_timeout), "bash", "-s"],
            input=command,
            capture_output=True,
            text=True,
            errors="replace",
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 124:
            output += f"\n<timeout>Command killed after {self.cfg.command_timeout}s</timeout>"
        return result.returncode, output

    def stop(self) -> None:
        if self.started:
            subprocess.run(["docker", "rm", "--force", self.name], capture_output=True)
            self.started = False


def run_command(command: str, cfg: Config, sandbox: Sandbox | None = None) -> tuple[int, str]:
    """Run one command in a subshell, exactly as the training environment did.

    Each command gets a fresh subshell (cd and exports do not persist) and its own
    process group, so a timeout or Ctrl-C kills the whole tree rather than orphaning
    children that keep holding the terminal.
    """
    if sandbox is not None:
        try:
            return sandbox.run(command)
        except KeyboardInterrupt:
            return 130, "<interrupted>The user interrupted this command.</interrupted>"
    env = {
        **os.environ,
        "PAGER": "cat",
        "MANPAGER": "cat",
        "GIT_PAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
        "CI": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "DEBIAN_FRONTEND": "noninteractive",
    }
    try:
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=cfg.cwd,
            env=env,
            stdin=subprocess.DEVNULL,   # no command may block waiting for input
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            errors="replace",
        )
    except OSError as error:
        return 1, f"Failed to start the command: {error}"

    def kill_tree():
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()

    try:
        output, _ = process.communicate(timeout=cfg.command_timeout)
        return process.returncode, output
    except subprocess.TimeoutExpired:
        kill_tree()
        output, _ = process.communicate()
        return 124, (output or "") + f"\n<timeout>Command killed after {cfg.command_timeout}s</timeout>"
    except KeyboardInterrupt:
        kill_tree()
        process.communicate()
        return 130, "<interrupted>The user interrupted this command.</interrupted>"


# --------------------------------------------------------------------------- approval

def prefilled_input(prompt: str, text: str) -> str:
    """input() with `text` already on the line, so an edit starts from the command."""
    try:
        import readline
    except ImportError:
        return input(prompt) or text

    def hook():
        readline.insert_text(text)
        readline.redisplay()

    readline.set_startup_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


def approve(command: str, cfg: Config) -> tuple[str, str]:
    """Ask before running. Returns (decision, command); decision is run/skip/quit/always."""
    danger = risk_of(command, sandboxed=bool(cfg.docker_image))
    if cfg.auto_approve and not danger:
        return "run", command
    if danger:
        print(S.red(f"  ! {danger} — review this one carefully"))
    if not sys.stdin.isatty():
        # Nobody is there to answer, so refusing beats running blind.
        if danger:
            print(S.yellow("  declined: no terminal to confirm on"))
            return "skip", command
        return "run", command
    options = "[y] run  [n] skip  [e] edit  [a] run all  [q] quit"
    while True:
        try:
            answer = input(f"  {S.prompt('run it?', '1')} {S.prompt(options, '2')} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "quit", command
        if answer in ("", "y", "yes"):
            return "run", command
        if answer in ("n", "no", "s", "skip"):
            return "skip", command
        if answer in ("q", "quit", "exit"):
            return "quit", command
        if answer in ("a", "all", "always"):
            return "always", command
        if answer in ("e", "edit"):
            edited = prefilled_input("  " + S.prompt("$ ", "36"), command).strip()
            if edited and edited != command:
                # Readline prefill means a hasty edit can append to the original, so
                # show exactly what is about to run.
                print(S.dim("  running: ") + S.bold(edited.replace("\n", S.dim("\n  | "))))
                return "run", edited
            if edited:
                return "run", edited


# --------------------------------------------------------------------------- context

def estimate_tokens(messages: list[dict]) -> int:
    # ~3.5 chars/token for code and shell output; close enough to decide when to elide.
    return int(sum(len(m["content"]) for m in messages) / 3.5) + 8 * len(messages)


ELIDED = "<output>[earlier output elided to free context]</output>"
TRUNCATED = "\n<elided>[…truncated to fit the context window]</elided>"

CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length|context[_ ]length_exceeded|longer than the maximum|"
    r"reduce the length|decrease the (input|prompt) length",
    re.IGNORECASE,
)


def trim_history(messages: list[dict], cfg: Config, keep: int = 6) -> list[dict]:
    """Elide the oldest observations when the conversation nears the context window.

    The system prompt, the task, and the most recent steps are what the model needs;
    old command output is the cheapest thing to drop. Dropping whole messages would
    break the user/assistant alternation the model was trained on, so observations are
    replaced in place instead.

    `keep` is how many trailing messages stay untouched. The caller lowers it to retry
    after the server rejects a prompt as too long.
    """
    budget = cfg.max_context_tokens - cfg.max_tokens - 512
    if budget <= 0 or estimate_tokens(messages) <= budget:
        return messages

    trimmed = list(messages)
    # Index 0 is the system prompt and index 1 the task; both always stay.
    tail = max(len(trimmed) - keep, 2)
    for index in range(2, tail):
        if trimmed[index]["role"] != "user" or trimmed[index]["content"] == ELIDED:
            continue
        trimmed[index] = {"role": "user", "content": ELIDED}
        if estimate_tokens(trimmed) <= budget:
            return trimmed

    # Everything old is already elided and it still does not fit, so start cutting the
    # recent observations down rather than letting the server reject the whole prompt.
    cap = 4000
    while cap >= 250 and estimate_tokens(trimmed) > budget:
        for index in range(2, len(trimmed)):
            content = trimmed[index]["content"]
            if trimmed[index]["role"] == "user" and len(content) > cap:
                trimmed[index] = {"role": "user", "content": content[:cap] + TRUNCATED}
        cap //= 2
    return trimmed


# --------------------------------------------------------------------------- agent

class Agent:
    """The mini-swe-agent loop: ask the model, run its command, show it the result."""

    def __init__(self, cfg: Config, sandbox: Sandbox | None = None):
        self.cfg = cfg
        self.sandbox = sandbox
        self.messages: list[dict] = []
        self.steps = 0
        self.last_command = ""
        self.repeats = 0
        self.last_response = ""
        self.response_repeats = 0
        self.format_errors = 0

    def start(self, task: str) -> None:
        sandbox_note = self.sandbox.note() if self.sandbox is not None else ""
        self.messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {
                "role": "user",
                "content": INSTANCE_TEMPLATE.format(
                    task=task.strip(),
                    cwd=self.cfg.cwd,
                    platform=f"{os.uname().sysname} {os.uname().machine}",
                    sandbox=sandbox_note,
                    marker=SUBMIT_MARKER,
                ),
            },
        ]
        self.steps = 0
        self.last_command, self.repeats = "", 0
        self.last_response, self.response_repeats, self.format_errors = "", 0, 0

    def follow_up(self, task: str) -> None:
        """Add a new request to an existing conversation, keeping what came before.

        A session is a conversation: "now add tests for it" has to be able to mean the
        file the agent just wrote.
        """
        if not self.messages:
            self.start(task)
            return
        if self.messages[-1]["role"] != "assistant":
            # Never two user turns in a row: the fine-tune only ever saw strict
            # user/assistant alternation.
            self.messages.append({"role": "assistant", "content": "Understood."})
        self.messages.append({"role": "user", "content": FOLLOW_UP_TEMPLATE.format(task=task.strip())})
        self.steps = 0
        self.last_command, self.repeats = "", 0
        self.last_response, self.response_repeats, self.format_errors = "", 0, 0

    def _query(self) -> tuple[str, str]:
        """One model turn, streamed but stopping the display at the code fence."""
        state = {"shown": 0, "buffer": "", "opened": False}

        def on_text(piece: str) -> None:
            state["buffer"] += piece
            visible = strip_reasoning(state["buffer"])
            fence = visible.find("```")
            if fence != -1:
                state["opened"] = True
                visible = visible[:fence]
            # Hold back a tag that is still arriving: "<exam" must not reach the screen
            # before it can be recognised as scaffolding and dropped.
            opening = visible.rfind("<")
            if opening != -1 and ">" not in visible[opening:]:
                visible = visible[:opening]
            # Hold back trailing whitespace too: it is either the blank line before the
            # code fence (never worth printing) or mid-thought spacing that the next
            # chunk will flush along with the text that follows it.
            visible = visible.rstrip()
            if len(visible) > state["shown"]:
                text = visible[state["shown"]:]
                if state["shown"] == 0:
                    text = text.lstrip()
                    if text.upper().startswith("THOUGHT:"):
                        text = text[len("THOUGHT:"):].lstrip()
                    text = "  " + text
                # Keep the thought in the same two-space gutter as commands and output.
                sys.stdout.write(text.replace("\n", "\n  "))
                sys.stdout.flush()
                state["shown"] = len(visible)

        started = time.monotonic()
        for keep in (6, 2, 0):
            self.messages = trim_history(self.messages, self.cfg, keep=keep)
            try:
                text, finish_reason = complete(self.cfg, self.messages, on_text=on_text)
                break
            except ServerError as error:
                # The token estimate is approximate, so the server is the real judge of
                # what fits. Free more context and try again before giving up the task.
                if keep == 0 or not CONTEXT_OVERFLOW_RE.search(str(error)):
                    raise
                print(S.yellow("\n  prompt too long for the context window; "
                               "eliding earlier output and retrying"))
                state.update(shown=0, buffer="", opened=False)
        if not state["shown"] and not self.cfg.stream:
            thought = split_thought(text)
            if thought:
                sys.stdout.write("  " + thought.replace("\n", "\n  "))
                state["shown"] = len(thought)
        print(S.dim(f"  ({time.monotonic() - started:.1f}s)") if state["shown"]
              else S.dim(f"({time.monotonic() - started:.1f}s)"))
        return text, finish_reason

    def step(self) -> str:
        """Run one think-act-observe cycle. Returns continue/done/quit/limit."""
        self.steps += 1
        print()
        print(rule(f"step {self.steps}/{self.cfg.step_limit}"))
        try:
            text, finish_reason = self._query()
        except KeyboardInterrupt:
            print(S.yellow("\n  interrupted"))
            return "quit"
        self.messages.append({"role": "assistant", "content": text or ""})

        answer = strip_reasoning(text).strip()
        self.response_repeats = self.response_repeats + 1 if answer == self.last_response else 1
        self.last_response = answer
        if self.response_repeats >= STOP_AFTER:
            print(S.red(f"  stuck: {self.response_repeats} identical responses in a row, giving up"))
            return "stuck"

        actions = parse_actions(text)
        if not actions:
            if finish_reason == "length" and self.format_errors < FORMAT_RETRIES:
                # Cut off mid-sentence, so this is an unfinished turn, not an answer.
                self.format_errors += 1
                print(S.yellow("  response was cut off before a command; asking again"))
                self.messages.append({"role": "user", "content": LENGTH_ERROR})
                return "continue"
            if answer:
                # A reply with no command is the model answering directly, which is the
                # right move for a question that needs no shell at all.
                print(S.green("  ✓ answered"))
                return "done"
            print(S.yellow("  empty response; asking again"))
            self.messages.append({
                "role": "user",
                "content": FORMAT_ERROR_TEMPLATE.format(
                    error="Your response was empty.", count=0),
            })
            return "continue"
        if len(actions) > 1 and self.format_errors < FORMAT_RETRIES:
            self.format_errors += 1
            print(S.yellow(f"  {len(actions)} bash blocks in one response; asking for one"))
            self.messages.append({
                "role": "user",
                "content": FORMAT_ERROR_TEMPLATE.format(
                    error=f"Found {len(actions)} bash code blocks, expected exactly 1.",
                    count=len(actions),
                ),
            })
            return "continue"
        if len(actions) > 1:
            # Asking did not work, so take the first block and keep moving. The next
            # step gets the result and can run whatever was in the others.
            print(S.yellow(f"  still {len(actions)} blocks; running the first one only"))
        self.format_errors = 0

        command = actions[0]
        if is_submit(command):
            # Not executed: the trained submit command can carry `git add -A`, which
            # would stage the user's files. The marker alone means the agent is done.
            print(S.green("  ✓ agent finished"))
            return "done"

        print(S.cyan("  $ ") + S.bold(command.replace("\n", S.dim("\n  | "))))
        decision, command = approve(command, self.cfg)
        if decision == "quit":
            return "quit"
        if decision == "always":
            self.cfg.auto_approve = True
        if decision == "skip":
            self.messages.append({
                "role": "user",
                "content": "<returncode>1</returncode>\n<output>\nThe user declined to run that "
                           "command. Try a different approach, or ask the user what to do instead.\n</output>",
            })
            print(S.yellow("  skipped"))
            return "continue"

        self.repeats = self.repeats + 1 if command == self.last_command else 1
        self.last_command = command

        returncode, output = run_command(command, self.cfg, self.sandbox)
        shown = output.strip()
        if len(shown) > DISPLAY_CHAR_LIMIT:
            shown = shown[:DISPLAY_CHAR_LIMIT] + S.dim(f"\n  … {len(output) - DISPLAY_CHAR_LIMIT} more characters")
        if shown:
            print("\n".join("  " + line for line in shown.splitlines()))
        marker = S.green("exit 0") if returncode == 0 else S.red(f"exit {returncode}")
        print(S.dim("  → ") + marker)

        observation = format_observation(returncode, output)
        if self.repeats >= NUDGE_AFTER:
            print(S.yellow(f"  same command {self.repeats}× in a row; nudging it to change course"))
            observation += REPEAT_NUDGE.format(count=self.repeats)
        self.messages.append({"role": "user", "content": observation})

        if returncode == 130:
            return "quit"
        if self.repeats >= STOP_AFTER:
            print(S.red(f"  stuck: {self.repeats} identical commands in a row, giving up on this task"))
            return "stuck"
        return "continue"

    def run(self, task: str, keep_history: bool = False) -> str:
        if keep_history:
            self.follow_up(task)
        else:
            self.start(task)
        while self.steps < self.cfg.step_limit:
            try:
                status = self.step()
            except ServerError as error:
                print(S.red(f"\n{error}"))
                return "error"
            if status in ("done", "quit", "stuck"):
                return status
        print(S.yellow(f"\n  reached the {self.cfg.step_limit}-step limit; "
                       "rerun with --step-limit to give it more room"))
        return "limit"

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"model": self.cfg.model, "cwd": self.cfg.cwd, "steps": self.steps,
             "messages": self.messages},
            indent=2,
        ))
        print(S.dim(f"  transcript → {path}"))


# --------------------------------------------------------------------------- session

def show_changes(cfg: Config) -> None:
    """Read-only summary of what the agent touched, if this is a git repo."""
    probe = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        cwd=cfg.cwd, capture_output=True, text=True,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        print(rule("changes"))
        print("\n".join("  " + line for line in probe.stdout.strip().splitlines()))
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=cfg.cwd, capture_output=True, text=True,
    )
    if untracked.returncode == 0 and untracked.stdout.strip():
        print(S.dim("  new files: ") + ", ".join(untracked.stdout.split()[:10]))


HELP = """\
  <prompt>        give the agent a task, or ask it a question
  /new            forget the conversation and start fresh
  /auto           toggle running commands without asking
  /cwd [path]     show or change the working directory
  /steps N        set the per-task step limit
  /save [file]    write the conversation to JSON
  /model          show the server and model in use
  /help           this list
  /exit           quit"""


def interactive(cfg: Config, sandbox: Sandbox | None = None) -> int:
    try:
        import readline  # noqa: F401  (gives the prompt history and line editing)
        history = Path.home() / ".codesquire_history"
        try:
            readline.read_history_file(history)
        except (OSError, ValueError):
            pass
    except ImportError:
        history = None

    print_banner(cfg, sandbox)

    agent = Agent(cfg, sandbox)
    while True:
        try:
            line = input("\n" + S.prompt("› ", "35")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if history is not None:
            try:
                readline.write_history_file(history)
            except OSError:
                pass

        if line.startswith("/"):
            command, _, argument = line[1:].partition(" ")
            argument = argument.strip()
            if command in ("exit", "quit", "q"):
                break
            if command in ("help", "h", "?"):
                print(HELP)
            elif command in ("new", "reset", "clear"):
                agent = Agent(cfg, sandbox)
                print(S.dim("  new conversation"))
            elif command == "auto":
                cfg.auto_approve = not cfg.auto_approve
                print(S.dim("  auto-approve ") + (S.yellow("on") if cfg.auto_approve else S.green("off")))
            elif command == "cwd":
                if argument:
                    target = Path(argument).expanduser()
                    if target.is_dir():
                        cfg.cwd = str(target.resolve())
                    else:
                        print(S.red(f"  not a directory: {target}"))
                print(S.dim("  cwd ") + cfg.cwd)
            elif command == "steps":
                if argument.isdigit() and int(argument) > 0:
                    cfg.step_limit = int(argument)
                print(S.dim("  step limit ") + str(cfg.step_limit))
            elif command == "save":
                if agent.messages:
                    agent.save(Path(argument or "codesquire-transcript.json"))
                else:
                    print(S.dim("  nothing to save yet"))
            elif command == "model":
                print(S.dim(f"  {cfg.model} @ {cfg.api_base}  (context {cfg.max_context_tokens})"))
            else:
                print(S.red(f"  unknown command /{command}") + S.dim("  — /help"))
            continue

        agent.run(line, keep_history=bool(agent.messages))
        show_changes(cfg)
    return 0


# --------------------------------------------------------------------------- cli

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codesquire",
        description="Prompt the CodeSquire agent served on vLLM and let it write code here.",
        epilog="With no PROMPT, starts an interactive session.",
    )
    parser.add_argument("prompt", nargs="*", help="the task for the agent")
    parser.add_argument("--api-base", default=None,
                        help="vLLM OpenAI endpoint (default: $CODESQUIRE_API_BASE, else "
                             "whichever of localhost:8080 and localhost:8000 answers)")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="bearer token, if the server needs one")
    parser.add_argument("-m", "--model", default="", help="model id (default: whatever the server serves)")
    parser.add_argument("-C", "--cwd", default=None, help="working directory for commands (default: here)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="run commands without asking (still asks for dangerous ones)")
    # A plain flag, not an optional-value one: `--docker "fix the bug"` must not read the
    # prompt as an image name.
    parser.add_argument("--docker", action="store_true",
                        help="run every command in a throwaway container that shares only "
                             "the working directory. Implies --yes, since the sandbox is "
                             "then the safety boundary")
    parser.add_argument("--docker-image", default="python:3.12-slim", metavar="IMAGE",
                        help="image for --docker (default: python:3.12-slim)")
    parser.add_argument("--confirm", action="store_true",
                        help="keep asking before each command even under --docker")
    parser.add_argument("--no-network", dest="network", action="store_false",
                        help="cut the sandbox off from the network")
    parser.add_argument("--step-limit", type=int, default=40, help="max think-act cycles per task (default: 40)")
    parser.add_argument("--timeout", type=int, default=120, help="per-command timeout in seconds (default: 120)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20, help="vLLM sampling; 0 disables")
    parser.add_argument("--max-tokens", type=int, default=2048, help="cap per model turn (default: 2048)")
    parser.add_argument("--context", type=int, default=0,
                        help="context window to budget for (default: ask the server)")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false", default=None,
                          help="tell a Qwen3 chat template to skip <think>")
    thinking.add_argument("--thinking", dest="thinking", action="store_true",
                          help="allow <think> blocks (stripped before parsing)")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="wait for whole responses")
    parser.add_argument("--save", metavar="FILE", help="write the transcript to FILE when done")
    parser.add_argument("--version", action="version", version=f"codesquire {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cwd = Path(args.cwd).expanduser() if args.cwd else Path.cwd()
    if not cwd.is_dir():
        print(S.red(f"Not a directory: {cwd}"), file=sys.stderr)
        return 2

    cfg = Config(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        step_limit=args.step_limit,
        command_timeout=args.timeout,
        cwd=str(cwd.resolve()),
        auto_approve=args.yes or (args.docker and not args.confirm),
        stream=args.stream,
        docker_image=args.docker_image if args.docker else "",
        network=args.network,
    )
    if args.confirm:
        cfg.auto_approve = False

    candidates = [args.api_base] if args.api_base else default_endpoints()
    try:
        cfg, model_id, served_length = resolve_endpoint(cfg, candidates)
    except ServerError as error:
        print(S.red(str(error)), file=sys.stderr)
        return 1
    cfg = replace(
        cfg,
        model=model_id,
        max_context_tokens=args.context or served_length or cfg.max_context_tokens,
    )

    sandbox = None
    if cfg.docker_image:
        sandbox = Sandbox(cfg)
        try:
            print(S.dim(f"  starting sandbox: {cfg.docker_image}"), flush=True)
            sandbox.start()
        except RuntimeError as error:
            print(S.red(str(error)), file=sys.stderr)
            return 1

    try:
        return _dispatch(cfg, args, sandbox)
    finally:
        if sandbox is not None:
            sandbox.stop()


def _dispatch(cfg: Config, args: argparse.Namespace, sandbox: Sandbox | None) -> int:
    task = " ".join(args.prompt).strip()
    if not task:
        if not sys.stdin.isatty():
            task = sys.stdin.read().strip()      # `echo "fix the test" | codesquire`
        if not task:
            return interactive(cfg, sandbox)

    if not sys.stdin.isatty() and not cfg.auto_approve:
        # Without a terminal there is nobody to approve each command, and running the
        # agent's commands unattended has to be a deliberate choice.
        print(S.red("No terminal to confirm commands on. Pass --yes to let the agent run "
                    "them unattended, or start codesquire from an interactive shell."),
              file=sys.stderr)
        return 2

    agent = Agent(cfg, sandbox)
    status = agent.run(task)
    show_changes(cfg)
    if args.save:
        agent.save(Path(args.save))
    return 0 if status == "done" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
