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

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
