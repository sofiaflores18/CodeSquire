#!/usr/bin/env python
"""`mini-extra swebench` with two fixes, used by run_eval.py. Same CLI flags.

1. Sandbox cleanup: the stock runner never calls env.stop(), so every task's
   Modal sandbox stays alive (and billed) until its runtime timeout.
2. Early stop for loops: small models often rerun one command forever after
   it returns nothing. After STOP_AFTER identical consecutive steps the task
   ends as a failure with exit status RepeatedCommandLoop. The model is never
   told about this (no nudge), so it scores the same as hitting the step
   limit; it just stops wasting ~90 calls per stuck task.
"""
import json

from minisweagent.exceptions import LimitsExceeded
from minisweagent.run.benchmarks import swebench

STOP_AFTER = 8  # identical consecutive steps before ending the task


class GuardedAgent(swebench.ProgressTrackingAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_actions = None
        self._repeats = 0

    def execute_actions(self, message: dict) -> list[dict]:
        actions = json.dumps(message.get("extra", {}).get("actions", []), sort_keys=True)
        self._repeats = self._repeats + 1 if actions == self._last_actions else 1
        self._last_actions = actions

        observations = super().execute_actions(message)
        if self._repeats >= STOP_AFTER:
            raise LimitsExceeded({
                "role": "exit",
                "content": "RepeatedCommandLoop",
                "extra": {"exit_status": "RepeatedCommandLoop", "submission": ""},
            })
        return observations

    def run(self, *args, **kwargs) -> dict:
        try:
            return super().run(*args, **kwargs)
        finally:
            if hasattr(self.env, "stop"):
                self.env.stop()


swebench.ProgressTrackingAgent = GuardedAgent

if __name__ == "__main__":
    swebench.app()
