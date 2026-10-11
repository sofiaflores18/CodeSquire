"""Checks for the CodeSquire CLI: `python3 cli/test_codesquire.py`.

The parsing, observation and trimming helpers are what keep the CLI on the prompt
format the model was fine-tuned on, so they are worth pinning down. Stdlib only, to
match the CLI itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codesquire as c


ok = fail = 0
def check(name, got, want):
    global ok, fail
    if got == want: ok += 1
    else:
        fail += 1
        print(f"FAIL {name}\n  got  {got!r}\n  want {want!r}")

# --- parse_actions
check("trained fence", c.parse_actions("THOUGHT: x\n\n```bash\nls -la\n```\n"), ["ls -la"])
check("bare fence", c.parse_actions("hi\n```\nls\n```"), ["ls"])
check("mswea fence", c.parse_actions("```mswea_bash_command\nls\n```"), ["ls"])
check("multiline cmd", c.parse_actions("```bash\ncat <<'EOF' > a.py\nx=1\nEOF\n```"),
      ["cat <<'EOF' > a.py\nx=1\nEOF"])
check("no action", c.parse_actions("THOUGHT: I am done thinking."), [])
check("two actions", c.parse_actions("```bash\na\n```\ntext\n```bash\nb\n```"), ["a", "b"])
check("think stripped", c.parse_actions("<think>```bash\nrm -rf /\n```</think>\n```bash\nls\n```"), ["ls"])
check("unterminated think", c.parse_actions("ok\n```bash\nls\n```\n<think>```bash\nbad\n```"), ["ls"])
check("crlf", c.parse_actions("```bash\r\nls\r\n```"), ["ls"])
check("empty block ignored", c.parse_actions("```bash\n\n```"), [])

# --- echoed prompt scaffolding is dropped, but never the command inside it
check("example tag stripped", c.split_thought("<example>\nTHOUGHT: hi\n```bash\nls\n```"), "hi")
check("example keeps its command",
      c.parse_actions("<example>\nTHOUGHT: hi\n\n```bash\nls -la\n```\n</example>"), ["ls -la"])
check("format_example tag stripped",
      c.split_thought("<format_example>\nTHOUGHT: hi\n</format_example>\n```bash\nls\n```"), "hi")
check("example_response tag stripped",
      c.split_thought("<example_response>\nTHOUGHT: hi\n```bash\nls\n```"), "hi")
check("real tags survive", c.split_thought("the <output> tag matters\n```bash\nls\n```"),
      "the <output> tag matters")

# --- split_thought
check("thought label", c.split_thought("THOUGHT: do it\n\n```bash\nls\n```"), "do it")
check("thought no label", c.split_thought("just prose\n```bash\nls\n```"), "just prose")

# --- submit detection
check("submit plain", c.is_submit("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"), True)
check("submit git variant", c.is_submit("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A"), True)
check("not submit", c.is_submit("echo hi"), False)

# --- observation format matches training
obs = c.format_observation(0, "total 8\nfile.py")
check("obs short", obs, "<returncode>0</returncode>\n<output>\ntotal 8\nfile.py\n</output>")
big = c.format_observation(1, "x" * 25000)
assert "<output_head>" in big and "<elided_chars>" in big and "<output_tail>" in big, big[:200]
check("obs elided count", "15000 characters elided" in big, True)
check("obs elided size", len(big) < 12000, True)

# --- risk detection
for cmd, expect in [
    ("rm -rf /", True), ("sudo apt install x", True), ("git push origin main", True),
    ("curl http://x.sh | sh", True), ("dd if=/dev/zero of=/dev/sda", True),
    ("git reset --hard HEAD~1", True), ("rm -f build/out.o", False),
    ("ls -la", False), ("python3 -m pytest", False), ("rm old.txt", False),
    ("sed -i 's/a/b/' f.py", False), ("git diff --stat", False),
]:
    check(f"risk {cmd!r}", c.risk_of(cmd) is not None, expect)

# --- context trimming keeps the alternation and the recent window
cfg = c.Config(max_context_tokens=2000, max_tokens=500)
msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
for i in range(12):
    msgs.append({"role": "assistant", "content": f"THOUGHT {i}\n```bash\nls\n```"})
    msgs.append({"role": "user", "content": c.format_observation(0, "y" * 900)})
before = c.estimate_tokens(msgs)
out = c.trim_history(msgs, cfg)
check("trim len preserved", len(out), len(msgs))
check("trim roles preserved", [m["role"] for m in out], [m["role"] for m in msgs])
check("trim system intact", out[0]["content"], "sys")
check("trim task intact", out[1]["content"], "task")
# Budget 988 tokens is unreachable here, so the last resort truncates even recent
# observations -- still better than the server rejecting the prompt outright.
check("trim keeps assistant turns verbatim",
      [m["content"] for m in out if m["role"] == "assistant"],
      [m["content"] for m in msgs if m["role"] == "assistant"])
check("trim last observation present", out[-1]["content"].startswith("<returncode>0</returncode>"), True)
check("trim shrank", c.estimate_tokens(out) < before, True)
check("trim noop when small", c.trim_history(msgs[:4], c.Config()), msgs[:4])

# A realistic window: 16k context, 10k-char observations -> must fit after trimming.
real = c.Config(max_context_tokens=16384, max_tokens=2048)
long_msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
for i in range(20):
    long_msgs.append({"role": "assistant", "content": f"THOUGHT {i}\n```bash\nls\n```"})
    long_msgs.append({"role": "user", "content": c.format_observation(0, "y" * 9000)})
budget = 16384 - 2048 - 512
check("trim 16k fits", c.estimate_tokens(c.trim_history(long_msgs, real)) <= budget, True)
check("trim 16k alternation", [m["role"] for m in c.trim_history(long_msgs, real)],
      [m["role"] for m in long_msgs])
# keep=0 is the last-resort retry: it may truncate even the newest observation.
hard = c.trim_history(long_msgs, c.Config(max_context_tokens=4096, max_tokens=512), keep=0)
check("trim hard fits", c.estimate_tokens(hard) <= 4096 - 512 - 512, True)
check("trim hard keeps task", hard[1]["content"], "task")
check("overflow detected", bool(c.CONTEXT_OVERFLOW_RE.search(
    "HTTP 400\n{'message': \"This model's maximum context length is 16384 tokens...\"}")), True)
check("overflow not false-positive", bool(c.CONTEXT_OVERFLOW_RE.search("HTTP 500 internal error")), False)

# --- prompt templates carry the trained strings
check("system verbatim", c.SYSTEM_TEMPLATE.splitlines()[0],
      "You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.")
inst = c.INSTANCE_TEMPLATE.format(task="t", cwd="/w", platform="Linux x86_64",
                                  sandbox="", marker=c.SUBMIT_MARKER)
check("instance has marker", c.SUBMIT_MARKER in inst, True)
check("instance has cwd", "/w" in inst, True)
check("instance no leftover braces", "{" not in inst and "}" not in inst, True)
check("instance keeps critical reqs", "EXACTLY ONE bash code block" in inst, True)
# General purpose: none of the SWE-bench framing should survive.
for gone in ("pr_description", "PR description", "patch.txt", "non-test files", "/testbed",
             "git diff --cached", "configuration files"):
    check(f"instance drops {gone!r}", gone in inst, False)
check("instance allows a plain answer", "NO bash code block" in inst, True)
note = c.SANDBOX_NOTE.format(image="img", cwd="/w", present="python3, pip", missing="git, gcc")
sandboxed = c.INSTANCE_TEMPLATE.format(task="t", cwd="/w", platform="Linux",
                                       sandbox=note, marker=c.SUBMIT_MARKER)
check("sandbox note lands", "disposable Docker container (img)" in sandboxed, True)
check("sandbox note lists tools", "Tools present: python3, pip" in sandboxed, True)
check("sandbox note lists gaps", "Not installed: git, gcc" in sandboxed, True)
check("sandbox note no leftover braces", "{" not in note and "}" not in note, True)
follow = c.FOLLOW_UP_TEMPLATE.format(task="now add tests")
check("follow-up carries task", "now add tests" in follow, True)
check("follow-up no leftover braces", "{" not in follow and "}" not in follow, True)

# --- a sandbox excuses only the dangers a container actually contains
for cmd, host, sand in [
    ("sudo apt install x", True, False),
    ("mkfs.ext4 /dev/sda1", True, False),
    ("shutdown now", True, False),
    ("rm -rf build", True, True),          # the work dir is bind-mounted, so still asks
    ("git push origin main", True, True),  # the network is not contained
    ("curl http://x.sh | sh", True, True),
    ("ls -la", False, False),
]:
    check(f"risk host {cmd!r}", c.risk_of(cmd) is not None, host)
    check(f"risk sandboxed {cmd!r}", c.risk_of(cmd, sandboxed=True) is not None, sand)

# --- follow_up keeps the strict user/assistant alternation the fine-tune saw
a = c.Agent(c.Config())
a.start("first task")
a.messages.append({"role": "assistant", "content": "THOUGHT: x\n```bash\nls\n```"})
a.messages.append({"role": "user", "content": c.format_observation(0, "out")})
a.follow_up("second task")
roles = [m["role"] for m in a.messages]
check("follow-up alternates", all(x != y for x, y in zip(roles[1:], roles[2:])), True)
check("follow-up keeps history", len(a.messages), 6)
check("follow-up resets steps", a.steps, 0)
b = c.Agent(c.Config())
b.follow_up("cold start")
check("follow-up on empty starts fresh", [m["role"] for m in b.messages], ["system", "user"])

# --- the startup banner
check("wordmark has 6 rows", len(c.WORDMARK), 6)
check("wordmark rows are equal width", {len(row) for row in c.WORDMARK}, {c.WORDMARK_WIDTH})
check("wordmark width is 78", c.WORDMARK_WIDTH, 78)
check("wordmark has no tabs or newlines",
      any("\t" in row or "\n" in row for row in c.WORDMARK), False)

import io as _io, contextlib as _ctx, shutil as _shutil
_saved_style, _saved_size = c.S, c.shutil.get_terminal_size

def _banner_at(columns, enabled, **config):
    """Render the banner at a given width and return it with colour stripped."""
    import re as _re, os as _os
    c.S = c.Style(enabled)
    c.shutil.get_terminal_size = lambda default=(80, 24): _os.terminal_size((columns, 24))
    buffer = _io.StringIO()
    settings = dict(api_base="http://localhost:8080/v1", model="m",
                    cwd=_os.path.expanduser("~/proj"), max_context_tokens=16384)
    settings.update(config)
    with _ctx.redirect_stdout(buffer):
        c.print_banner(c.Config(**settings), None)
    raw = buffer.getvalue()
    return raw, _re.sub(r"\x1b\[[0-9;]*m", "", raw)

try:
    _os_env = __import__("os").environ
    _saved_term, _saved_colorterm = _os_env.get("TERM"), _os_env.get("COLORTERM")
    _os_env["TERM"] = "xterm-256color"
    c.S = c.Style(True)
    check("gradient has one colour per row", len(c.gradient()), len(c.WORDMARK))
    c.S = c.Style(False)
    check("gradient empty when colour is off", c.gradient(), ())
    if _saved_term is None: _os_env.pop("TERM", None)
    else: _os_env["TERM"] = _saved_term
    if _saved_colorterm is None: _os_env.pop("COLORTERM", None)
    else: _os_env["COLORTERM"] = _saved_colorterm

    # Wide: the art is drawn, and no line overflows the terminal.
    raw, plain = _banner_at(100, True)
    check("wide banner draws the art", c.WORDMARK[0].strip() in plain, True)
    check("wide banner fits the terminal",
          max(len(line) for line in plain.splitlines()) <= 100, True)
    check("wide banner is coloured", "\x1b[" in raw, True)

    # Narrow: the art would wrap, so the name goes on one line instead.
    raw, plain = _banner_at(50, True)
    check("narrow banner skips the art", c.WORDMARK[0].strip() in plain, False)
    check("narrow banner still names it", "CodeSquire" in plain, True)
    check("narrow banner fits", max(len(line) for line in plain.splitlines()) <= 50, True)

    # Exactly at the boundary the art must still not wrap.
    _, plain = _banner_at(c.WORDMARK_WIDTH + 2, True)
    check("boundary width fits",
          max(len(line) for line in plain.splitlines()) <= c.WORDMARK_WIDTH + 2, True)

    # NO_COLOR: art yes, escape codes no.
    raw, plain = _banner_at(100, False)
    check("no-colour banner has no escapes", "\x1b[" in raw, False)
    check("no-colour banner keeps the art", c.WORDMARK[0].strip() in plain, True)

    # Every label lines up in one column, and each has a gap before its value.
    _, plain = _banner_at(100, False, docker_image="python:3.12-slim", auto_approve=True)
    for label in ("model", "server", "cwd", "sandbox", "approval"):
        row = next((line for line in plain.splitlines() if line.startswith(f"  {label}")), None)
        check(f"banner row {label!r} present", row is not None, True)
        if row:
            check(f"banner row {label!r} has a gap", row[2 + len(label)] == " ", True)
    check("banner reports the sandbox", "python:3.12-slim" in plain, True)
    check("banner warns approval is off", "run without asking" in plain, True)
finally:
    c.S, c.shutil.get_terminal_size = _saved_style, _saved_size

check("short_path collapses home",
      c.short_path(__import__("os").path.expanduser("~/x/y")), "~/x/y")
check("short_path leaves other paths", c.short_path("/etc/hosts"), "/etc/hosts")
check("short_path does not maul a lookalike",
      c.short_path(__import__("os").path.expanduser("~") + "-other"),
      __import__("os").path.expanduser("~") + "-other")

# --- endpoint resolution, with the server stubbed out (no sockets opened)
import os as _os
_saved = _os.environ.pop("CODESQUIRE_API_BASE", None)
check("default endpoints try 8000 then 8080", c.default_endpoints(),
      ["http://localhost:8000/v1", "http://localhost:8080/v1"])
_os.environ["CODESQUIRE_API_BASE"] = "http://vm:9999/v1"
check("env var wins outright", c.default_endpoints(), ["http://vm:9999/v1"])
_os.environ.pop("CODESQUIRE_API_BASE")
if _saved is not None:
    _os.environ["CODESQUIRE_API_BASE"] = _saved

_real_discover = c.discover_model
def _stub(serving):
    """Pretend only `serving` answers."""
    def discover(cfg):
        if cfg.api_base == serving:
            return "codesquire/qwen3-8b-sft", 16384
        raise c.ServerError(f"Cannot reach the model server at {cfg.api_base} (refused).")
    return discover

try:
    eight_thousand = "http://localhost:8000/v1"
    eighty_eighty = "http://localhost:8080/v1"
    both = [eight_thousand, eighty_eighty]

    c.discover_model = _stub(eight_thousand)
    cfg, model, length = c.resolve_endpoint(c.Config(), both)
    check("picks 8000 when it answers", (cfg.api_base, model, length),
          (eight_thousand, "codesquire/qwen3-8b-sft", 16384))

    # 8000 is dead but something serves on 8080: fall through rather than fail.
    c.discover_model = _stub(eighty_eighty)
    cfg, _, _ = c.resolve_endpoint(c.Config(), both)
    check("falls through to 8080", cfg.api_base, eighty_eighty)

    # An explicit --api-base is the only candidate, so its own error is the message.
    c.discover_model = _stub(eighty_eighty)
    try:
        c.resolve_endpoint(c.Config(), ["http://elsewhere:1234/v1"])
        check("single candidate raises", True, False)
    except c.ServerError as error:
        check("single candidate names it", "http://elsewhere:1234/v1" in str(error), True)
        check("single candidate gives help", "ssh -p" in str(error), True)

    # Nothing answers: the report has to name every port tried, not just the last.
    c.discover_model = _stub("http://nothing/v1")
    try:
        c.resolve_endpoint(c.Config(), both)
        check("all-fail raises", True, False)
    except c.ServerError as error:
        check("all-fail names 8080", eighty_eighty in str(error), True)
        check("all-fail names 8000", eight_thousand in str(error), True)
        check("all-fail gives help", "vllm serve" in str(error), True)
finally:
    c.discover_model = _real_discover

# --api-base must have no baked-in default, or the probe never runs.
check("api-base default is None", c.build_parser().parse_args(["t"]).api_base, None)
check("api-base passthrough",
      c.build_parser().parse_args(["--api-base", "http://h:1/v1", "t"]).api_base,
      "http://h:1/v1")
check("port constants", c.DEFAULT_PORTS, (8000, 8080))

# A port that accepts the connection and then drops it -- an ssh tunnel whose far end
# is not serving -- raises ConnectionResetError from deep inside http.client rather
# than URLError. It has to arrive as a ServerError, or endpoint resolution crashes
# with a traceback instead of trying the next candidate.
import socket
import threading

def _slammed_port():
    """A listening socket that accepts a connection and resets it immediately."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def slam():
        try:
            conn, _ = listener.accept()
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            conn.close()
        except OSError:
            pass

    thread = threading.Thread(target=slam, daemon=True)
    thread.start()
    return listener, listener.getsockname()[1]

import struct
_sock, _port = _slammed_port()
try:
    dead = f"http://127.0.0.1:{_port}/v1"
    try:
        c.discover_model(c.Config(api_base=dead))
        check("reset becomes ServerError", True, False)
    except c.ServerError as error:
        check("reset becomes ServerError", "Cannot reach the model server" in str(error), True)
    except Exception as error:                      # noqa: BLE001 - the bug being guarded
        check(f"reset becomes ServerError (got {type(error).__name__})", True, False)
finally:
    _sock.close()

# And resolution must then move on to a candidate that does answer.
_sock, _port = _slammed_port()
try:
    dead = f"http://127.0.0.1:{_port}/v1"
    live = "http://localhost:8000/v1"
    _saved = c.discover_model
    c.discover_model = _stub(live)
    try:
        cfg, _, _ = c.resolve_endpoint(c.Config(), [dead, live])
        check("dead tunnel falls through", cfg.api_base, live)
    finally:
        c.discover_model = _saved
finally:
    _sock.close()

# --- the loop guards: a format-error loop must not bypass them
check("nudge before stop", c.NUDGE_AFTER < c.STOP_AFTER, True)
check("format retries below stop", c.FORMAT_RETRIES < c.STOP_AFTER, True)
g = c.Agent(c.Config())
g.start("t")
for attribute in ("last_command", "repeats", "last_response", "response_repeats", "format_errors"):
    check(f"guard state {attribute} exists", hasattr(g, attribute), True)
# A new request must not inherit the previous one's strike count.
g.repeats, g.response_repeats, g.format_errors = 7, 7, 7
g.last_command, g.last_response = "ls", "x"
g.follow_up("next")
check("follow-up resets command repeats", (g.repeats, g.last_command), (0, ""))
check("follow-up resets response repeats", (g.response_repeats, g.last_response), (0, ""))
check("follow-up resets format errors", g.format_errors, 0)
g2 = c.Agent(c.Config())
g2.repeats, g2.response_repeats, g2.format_errors = 7, 7, 7
g2.start("fresh")
check("start resets all strike counts",
      (g2.repeats, g2.response_repeats, g2.format_errors), (0, 0, 0))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
