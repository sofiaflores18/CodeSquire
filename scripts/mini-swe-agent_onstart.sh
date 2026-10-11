cd /workspace
source .venv/bin/activate

export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://localhost:8000/v1
export MSWEA_COST_TRACKING=ignore_errors

mini \
  --model openai/Qwen/Qwen2.5-Coder-3B-Instruct \
  -c mini.yaml \
  -c model.model_kwargs.tool_choice=required
