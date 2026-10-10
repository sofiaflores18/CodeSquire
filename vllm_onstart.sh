
#!/usr/bin/env bash
set -e

cd /workspace

# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Create Python environment
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install mini-swe-agent
uv pip install litellm
uv pip install vllm

# 4. Configure mini-SWE-agent + LiteLLM
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://localhost:8000/v1
export MSWEA_COST_TRACKING=ignore_errors

# 5. Start vLLM
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"

vllm serve "$MODEL" \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
