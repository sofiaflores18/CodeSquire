# CodeSquire KH9 — GRPO training implementation context

## Objective and immediate priority

Build a working coding-agent training pipeline using GRPO and LoRA, then evaluate
the saved adapter through the existing inference/evaluation infrastructure.
The immediate milestone is one correct end-to-end optimizer update, a reloadable
checkpoint, and proof that subsequent rollouts use the updated policy. A long
training run and measurable improvement come after that milestone.

This context was prepared on October 10, 2026. Hacking ends October 11, 2026 at
11:00 AM America/New_York (EDT; 15:00 UTC). Prioritize the smallest working
implementation that preserves training correctness. Do not promise improvement
from a short run or describe an infrastructure smoke test as a benchmark gain.

## Known state versus assumptions

- The user now has access to a newly provisioned Vast.ai VM.
- The intended setup is a full Linux VM so Docker can run normally. Verify this
  and actual Docker operation; SSH access alone does not prove it is a full VM.
- The exact rented GPU model, count, per-device VRAM, CPU/RAM, disk, driver,
  template, repository path, and networking have NOT been supplied here.
- Discussed options included 2 x RTX 4090 (24GB each), 2 x RTX 5000 Ada (32GB
  each), and a listing described as 1 x RTX 4090 with 48GB. Do not assume which
  was rented. A standard 4090 has 24GB; validate a 48GB listing explicitly.
- A maintenance notice said October 11 to October 11. Exact time/timezone and
  rental expiration are unknown. Inspect them; do not assume availability
  through the deadline. Keep checkpoints and results backed up off-instance.
- The previous Vast container rental could not run Docker reliably. Modal was
  also problematic. The existing workaround generates patches remotely,
  transfers them through Git, and runs Docker grading locally.
- The repository inventory below came from the user. File implementations,
  dependency versions, benchmark scores, and successful model runs have not
  been inspected. Logs or filenames alone are not evidence of success.

## Existing project and repository inventory

The project uses mini-SWE-agent as the coding harness, vLLM for model serving,
and LiteLLM as a router where needed. SWE-bench supplies patch verification.
Model benchmarking has included several Qwen variants and VibeThinker.

User's local repository was ~/dev/CodeSquire. The VM checkout path is unknown.
Inventory reported before training implementation:

```text
configs/qwen.yaml
datasets/smoke_ids.txt
datasets/swebench_verified_validation_25_ids.txt
datasets/swebench_verified_validation_25.json
eval.py
gold.verified25_gold_check.json
mini-swe-agent_onstart.sh
models.yaml
README.md
scripts/overnight_eval.py
scripts/remote_eval.sh
scripts/run_swebench.sh
vllm_onstart.sh
logs/run_evaluation/...
overnight.log
overnight-logs/...
```

Preserve existing evaluation functionality. Read the actual files before
deciding what can be reused. Inspect model-request, tool-execution, trajectory
recording, patch-extraction, and grading boundaries first.

## Intended architecture

Prefer two GPUs in ONE VM, with separate responsibilities:

- Trainer GPU: Transformers/PyTorch model plus PEFT LoRA, optimizer, activations,
  and TRL GRPO optimization.
- Generation GPU: vLLM serving the current policy for agent rollouts.
- CPU: isolated repository environments, Docker tool execution and verification.
- Coordinator: groups attempts by issue/policy, captures traces, matches rewards,
  controls updates, synchronizes weights, and saves/resumes checkpoints.

There is no pooled VRAM requirement. Each GPU contains its own model copy and
must fit its assigned workload independently. No DDP, tensor parallelism, FSDP,
or DeepSpeed is needed merely to assign these two roles. Add sharding only if
measurement establishes a need. Set explicit CUDA_VISIBLE_DEVICES per process;
remember that visible physical GPU indices are renumbered inside that process.

Start with a synchronous loop:
1. Hold policy version v fixed and sample four independent attempts per issue.
2. Execute each attempt against the same clean base commit in its own workspace.
3. Verify final patches in separate clean grading environments.
4. Match rewards to their exact trajectories and compute group-relative updates.
5. Update LoRA and synchronize the resulting policy to the sampling engine.
6. Generate the next batch with the new version; checkpoint and log progress.

Two GPUs do not require asynchronous training or imply concurrent optimization
and sampling. Avoid stale-policy rollouts in the initial implementation. Do not
generate an entire static corpus and repeatedly call it fresh online GRPO data.

## Model and initial experiment

Intended first model: Qwen/Qwen3-4B-Instruct-2507, standard checkpoint.
Pin and record the exact base revision and tokenizer/chat template.

Starting choices, subject to profiling and pinned-library API constraints:
- BF16 frozen base with LoRA rank 16, alpha 32, dropout 0.
- Target supported attention and MLP linear modules; inspect actual module names
  or use supported all-linear targeting. Verify trainable parameter counts.
- Four generations/attempts per issue; they need not run concurrently.
- Training microbatch one, with valid generation-batch and gradient-accumulation
  settings that preserve complete groups. Four rollouts is not necessarily a
  per-device training batch of four.
- Gradient checkpointing and supported efficient attention.
- Start with about 4K total context per training sequence, then profile 8K.
  Context includes tool output and prior messages, not just generated tokens.
- Explicit generation, tool-output, agent-turn, command, and grading time limits.
- One optimization pass per fresh rollout batch initially.
- Decide and record KL/reference-policy behavior. Do not allocate an unnecessary
  standalone reference copy; do not assume beta=0 is appropriate without making
  the choice explicit. LoRA may allow a frozen-base reference via adapter disable.

Earlier planning estimates were ~14–24GB for the trainer and ~12–18GB for a
separate BF16 generation engine with short bounded contexts. These are NOT
measurements or guarantees. Profile a full generation and backward step with
realistic maximum-length traces. Large logits tensors and long agent histories
can invalidate these estimates. Consider QLoRA or reduced lengths/concurrency
if needed; do not assume an inference FP8 checkpoint is a drop-in training base.

## Critical integration: multi-turn agent trajectories

Choose one approach after inspecting mini-SWE-agent and installed TRL APIs:
1. Trainer owns the multi-turn loop through a stateful environment/tool wrapper.
2. mini-SWE-agent retains its loop; a compatible custom rollout/trace bridge
   captures its model calls and feeds them into the trainer.

Prefer retaining the existing harness if a correct supported bridge can be built
quickly. Do not replace the agent with single-response patch generation and
present that as equivalent. If using a simpler smoke task, label it clearly.
Do not assume a text transcript or final patch satisfies the trainer's rollout
contract. Inspect required token, log-probability, mask, and grouping semantics.

Capture task/run/batch/group/trajectory IDs, policy version, base commit, exact
per-turn prompt and completion token IDs, required behavior log probabilities,
tool observations, assistant-token loss masks, sampling parameters, termination,
patch, verifier status, and reward. Preserve exact sampling contexts, including
any context truncation. Do not retokenize a reconstructed conversation and
silently assume it matches the original engine tokens.

Optimize only generated policy tokens. User/system input, file content, tool
observations, and hidden grading data must not become generated-token targets.
Tool observations remain context. Apply terminal rewards to multi-turn traces
using the selected trainer's supported semantics; do not invent per-turn groups.
Ensure LiteLLM or another proxy does not discard required trace metadata.

## Data and reward correctness

- Keep datasets/swebench_verified_validation_25* held out from training.
  These are a project-selected subset of SWE-bench Verified, an evaluation set,
  not an official training split. Inspect IDs/content before making claims.
- Initially select 2–4 easy, disjoint training tasks with reproducible environments.
  Check task overlap and duplicate issues. Keep test/gold metadata verifier-side.
- Start with reward 1 for resolved under the official verifier and 0 for valid
  evaluation failures. Preserve regression checks, not just newly passing tests.
- Container/image/startup/verifier failures are infrastructure errors, not
  automatically reward 0. Retry or exclude according to a documented policy
  that preserves valid group handling. An agent exceeding its defined budget is
  different from the grading service failing.
- Grade extracted patches on clean bases. Isolate attempts and hidden grading
  artifacts; do not allow workspace state or gold solutions to leak across runs.
- Validate the verifier on unchanged and known-correct patches, separately from
  agent-visible training inputs.
- Monitor reward variance. Groups with identical rewards provide no relative
  task-reward signal. If everything fails, use easier tasks/exploration or discuss
  supervised warm-up; do not claim falling loss proves useful learning.

## Environment and implementation workflow

First inspect git status, existing instructions, README, scripts, and dependencies.
On the VM inspect nvidia-smi, virtualization, Docker hello-world, free RAM/disk,
driver and CUDA compatibility. Verify a real SWE-bench image as well.
Do not reinstall working host drivers or destroy existing user work casually.

Create a separate pinned training environment to avoid breaking evaluation.
TRL/vLLM APIs and weight-sync launch requirements vary by version. Use exact
version documentation, not guessed flags from a different release. Generic
OpenAI-compatible inference serving alone does not provide policy synchronization.
Use the supported server-mode weight-transfer path; establish whether it sends
adapters or merged weights. Avoid hand-writing synchronization unless necessary.

Suggested responsibilities (combine files if that speeds correct implementation):
- training/pyproject.toml and lockfile: dependencies.
- configs/grpo.yaml: model, LoRA, generation, reward, limits, saving.
- training/prepare_tasks.py: disjoint task preparation.
- training/environment.py: isolated repository execution.
- training/rollouts.py: grouped multi-turn generation and capture.
- training/rewards.py: clean verification and structured outcomes.
- training/train_grpo.py: trainer entry point and resume.
- scripts/train_remote.sh: GPU assignment, readiness, launch/logging.
- training/export.py: adapter/merged artifact for existing eval.

If Docker on this VM still fails, reuse local Docker as the execution/verifier
worker. Automate job/result transfer and preserve trajectory identities. SSH
queues or an authenticated API are preferable to per-update Git commits, but
the existing Git transport can serve a first proof. Remote grading does not by
itself solve Docker requirements for agent tool execution. Keep model service
reachable by the chosen worker and credentials out of logs/source control.

## Acceptance checks, in order

1. Actual hardware, VM, Docker, dependencies, and task environment verified.
2. Four independent attempts captured from one policy and clean task state.
3. Correct token eligibility masks, grouping, trace/reward matching, and no leaks.
4. Finite loss/gradients on a batch with reward variation; LoRA weights change
   after optimization and frozen base weights remain frozen.
5. Supported sync succeeds; inspect policy version and an engine-side weight or
   equivalent deterministic check. Randomly different text alone is not proof.
6. Save and reload adapter; test checkpoint resume including optimizer/trainer
   state as appropriate. Record exact model/tokenizer/config versions.
7. Run the exported checkpoint through existing evaluation; compare against the
   same base model with identical agent settings and held-out tasks.
8. Only then increase budget to a short 20–50-update run, with periodic saves.

Log reward/resolve rate, zero-variance groups, generated tokens, lengths, timeout
and infrastructure-error rates, rollout/grading/update timings, peak GPU memory,
gradient norm, and KL if used. Record seeds and policy/checkpoint versions.
Copy adapters/checkpoints and result summaries off the instance periodically,
especially before the unknown maintenance window. Save a demo-ready checkpoint
before spending the remaining time on larger experiments.

## Scope and collaboration

Implement the authorized training setup autonomously, making minimal changes.
Ask only for genuinely missing access/details that block dependent work.
Report what is verified versus planned. Preserve unrelated changes and existing
evaluation paths. Do not commit secrets, model weights, caches, or bulky rollout
logs to the source repository. Do not publish, spend on new rentals, destroy
instances, or send messages to others without user authorization.

Useful primary references (consult the documentation for pinned versions):
- https://huggingface.co/docs/trl/en/grpo_trainer
- https://huggingface.co/docs/trl/en/vllm_integration
- https://huggingface.co/docs/trl/en/reducing_memory_usage
- https://huggingface.co/docs/trl/main/openenv
- https://huggingface.co/docs/peft/en/package_reference/lora
- https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- https://www.swebench.com/SWE-bench/guides/evaluation/
- https://docs.vast.ai/guides/instances/virtual-machines
