# CodeSquire CLI

## Get started

```bash
# 1. Open a tunnel to the VM. Leave this terminal open.
ssh -p 30018 root@47.150.239.19 -L 8000:localhost:8000

# 2. In that same SSH session, serve the model. Leave it running.
#    scripts/vllm_onstart.sh does this for you, with these same settings.
vllm serve training/outputs --served-model-name qwen3-8b-sft \
  --host 127.0.0.1 --port 8000 --max-model-len 40960

# 3. In a NEW terminal on your machine, check the model is reachable.
#    Expect a JSON list with one model in it.
curl http://localhost:8000/v1/models

# 4. Still on your machine, go to the project you want it to work on and ask.
cd ~/your/project
~/PROJECTS/CodeSquire/cli/codesquire.py --docker "write a script that prints the first 10 primes, then run it"
```

That is everything. Step 4 needs no `--api-base`: with nothing specified the CLI tries
`localhost:8000` and then `localhost:8080`, so it finds the tunnel on its own.

Two things worth knowing about step 4: `--docker` runs the agent's commands in a
throwaway container that shares only the current directory, and the first run pulls
`python:3.12-slim` (~130 MB), so it takes a moment. Drop `--docker` to run commands
directly on your machine, where each one asks for approval first.

The one number that has to agree across steps 1-3 is the port. vLLM serves on 8000 (its
default, and what `scripts/vllm_onstart.sh` uses), so the tunnel forwards 8000 and the
curl checks 8000. If you serve on a different port, say `-L 8000:localhost:<that port>`
in step 1 and leave the rest alone — the tunnel is what translates, so the CLI still
finds it at `localhost:8000`.

If step 3 hangs or prints nothing, the tunnel is up but the far end is not serving: the
port in step 2 and the right-hand port in step 1 disagree. Check what the VM is actually
listening on with `ssh -p 30018 root@47.150.239.19 'ss -ltnp | grep vllm'`.

Optional, so you can type `codesquire` anywhere:

```bash
ln -s ~/PROJECTS/CodeSquire/cli/codesquire.py ~/.local/bin/codesquire
```

---

Prompt the CodeSquire agent from your terminal and let it work in your project. Ask it
anything — write a script, change some code, explain a codebase, run a command, answer a
question. The agent runs on the VM's vLLM server; this CLI is the other half of the loop
`run_eval.py` runs on SWE-bench: it reads the model's command, runs it, and feeds the
result back until the task is done.

`cli/codesquire.py` is stdlib-only and self-contained: copy it to any machine that can
reach the server.

## Setup

The tunnel in step 1 above is what keeps the model server off the public internet: vLLM
is started with `--host 127.0.0.1`, so it listens on the VM's loopback only and the
forward is the only way in. Run it in the background with `-N` if you would rather not
keep a shell open:

```bash
ssh -N -p 30018 root@47.150.239.19 -L 8000:localhost:8000 &
./cli/codesquire.py "add type hints to utils.py"
```

If vLLM is not installed on the VM yet, `scripts/vllm_onstart.sh` sets it up.

To reach a server on another host or port, pass `--api-base`, or set it once so you can
stop passing it:

```bash
set -Ux CODESQUIRE_API_BASE http://vm-host:8000/v1     # fish
export CODESQUIRE_API_BASE=http://vm-host:8000/v1      # bash/zsh
```

Pointing vLLM at the internet instead of a tunnel would expose an endpoint that runs
shell commands on request, so prefer the tunnel.

Put it on your PATH:

```bash
ln -s "$PWD/cli/codesquire.py" ~/.local/bin/codesquire
```

The model id comes from `/v1/models`, so there is nothing to keep in sync when you serve
a different checkpoint; the context budget comes from the server's `max_model_len`.

## Use

```bash
codesquire                                  # interactive session
codesquire "what does this project do?"
codesquire "write a script that dedupes lines in a CSV, then run it on sample.csv"
codesquire --docker "fix the failing test in test_api.py"
codesquire -C ~/some/project -y --step-limit 60 "migrate the config loader to pydantic"
echo "write a smoke test for the parser" | codesquire -y
```

Starting a session with no prompt opens on a banner showing the model, the endpoint, the
working directory and whether the sandbox is on, so what you are pointed at is on screen
before you type. It scales down on a narrow terminal and drops the colour under
`NO_COLOR`. One-shot prompts print nothing extra, so piping stays clean.

An interactive session is a conversation — "now add tests for that" means the file it
just wrote. `/new` starts over. Other commands: `/auto`, `/cwd [path]`, `/steps N`,
`/save [file]`, `/model`, `/help`, `/exit`.

A question that needs no shell gets answered directly, with no command run:

```
› what is the difference between a thread and a process?
  A process has its own memory space, while threads share one…
  ✓ answered
```

## Sandbox

`--docker` runs every command in a throwaway container instead of directly on your
machine. Recommended for a demo, and for any prompt you have not read first.

```bash
codesquire --docker "write a web scraper and try it on example.com"
codesquire --docker --docker-image python:3.12 "..."   # an image with git and gcc
codesquire --docker --no-network "..."                 # no network either
```

What the container shares with your computer is the working directory, bind-mounted at
the same absolute path so the model never has to translate paths. Everything else is
disposable: the host filesystem is not visible, the process count is capped, and files
the agent creates are owned by you, not root. One container lasts the session, so a
package installed in one step is still there in the next (`pip install --user` works).

The CLI probes the image at startup and tells the model which tools it actually has, so
it does not waste steps discovering that a slim image has no `git`.

Because the sandbox is then the safety boundary, `--docker` implies `--yes`. Pass
`--confirm` to keep being asked anyway.

## Approval

Without `--docker`, commands run on your machine, so each one is shown and waits:

```
  $ sed -i 's/old/new/' config.py
  run it? [y] run  [n] skip  [e] edit  [a] run all  [q] quit
```

`a` approves the rest of the task, `-y` approves from the start, `e` lets you edit the
command first. Commands that reach outside the project or cannot be undone (`sudo`,
`rm -rf`, `git push`, `curl | sh`, writes to a device) always ask, even under `-y`. Under
`--docker` the ones a container already contains stop asking, but the ones that reach the
mounted directory or the network still do. Running without a terminal requires `-y`, so
nothing executes unattended by accident.

| Flag | Default | Meaning |
|---|---|---|
| `--api-base` | `$CODESQUIRE_API_BASE`, else `localhost:8000` then `localhost:8080` | vLLM OpenAI endpoint |
| `-m`, `--model` | first model the server reports | model id |
| `-C`, `--cwd` | current directory | where commands run |
| `--docker` | off | run commands in a throwaway container; implies `-y` |
| `--docker-image` | `python:3.12-slim` | image for `--docker` |
| `--no-network` | network on | cut the sandbox off from the network |
| `-y`, `--yes` | off | run commands without asking |
| `--confirm` | — | keep asking even under `--docker` |
| `--step-limit` | 40 | think-act cycles per task |
| `--timeout` | 120 | per-command timeout, in seconds |
| `--temperature` / `--top-p` / `--top-k` | 0.7 / 0.8 / 20 | matches `configs/qwen3-8b.yaml` |
| `--max-tokens` | 2048 | cap per model turn |
| `--context` | server's `max_model_len` | context window to budget for |
| `--no-thinking` | unset | tells a Qwen3 chat template to skip `<think>` |
| `--no-stream` | off | wait for whole responses |
| `--save FILE` | — | write the conversation as JSON |

## How it matches the trained format

The agent was fine-tuned on `ricdomolm/mini-coder-trajs-400k`, so the CLI reproduces the
parts of that conversation shape the fine-tune actually conditioned on — change them and
the model goes off-distribution:

- the system prompt is verbatim from the training trajectories;
- the model replies with a `THOUGHT:` section and exactly one ` ```bash ` block;
- each result comes back as `<returncode>N</returncode>` plus `<output>`, with the same
  head/tail elision above 10,000 characters;
- turns strictly alternate user/assistant, including across follow-up requests;
- a malformed reply gets the trained format-error message rather than a lost step;
- a task ends when the model issues `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

What is deliberately *not* carried over is the SWE-bench framing. The task is a plain
`<task>` block instead of a PR description, there is no patch-building or
"don't touch the tests" scaffolding, and commands run in your working directory rather
than `/testbed`, so the model is free to answer whatever is asked. A reply with no
command at all is treated as a final answer, which the benchmark harness would have
rejected as a format error.

The submit command is never executed: the trained variant carries `git add -A` and would
stage your files. The marker alone ends the task.

Each command runs in a fresh subshell, so `cd` and exports do not persist between steps —
the same semantics the model was trained on. Prefix a command with `cd /path && ...` to
work elsewhere.

Three guards for small models. When a long task approaches the context window, the
oldest command outputs are elided in place (never dropped, which would break the
alternation); if the server still rejects the prompt, the CLI frees more context and
retries rather than losing the task.

When the model runs the same command over and over, it is nudged to change course after
3 repeats, and a task that produces 8 identical responses in a row is abandoned.
`scripts/swebench_runner.py` does the same for evals, but silently and only on repeated
commands, since a benchmark wants the honest failure and a person at a terminal wants a
way out.

And a malformed response is asked about twice, then worked with: a model that keeps
sending two bash blocks has the first one run rather than being refused forever. This
matters more than it sounds — repeats are counted on whole responses, not just on
commands that reached the shell, because a model stuck on formatting never gets as far
as running anything.

## Tests

```bash
python3 cli/test_codesquire.py
```

141 checks over the parsing, prompt, observation, trimming, risk and sandbox helpers. The
agent loop itself is exercised by hand against a mock vLLM server and a live one.
