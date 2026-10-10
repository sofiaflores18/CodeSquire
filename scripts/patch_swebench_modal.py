#!/usr/bin/env python3
"""Repair the Modal grading path in swebench >=5.

swebench 5.x moved to prebuilt per-instance evaluation images, published in the
dataset's `image` column, and dropped `setup_env_script` / `install_repo_script`
from TestSpec. Its Modal backend was not migrated and still tries to rebuild the
environment from scratch, so `ModalSandboxRuntime.get_instance_image` raises
AttributeError before any sandbox starts. swebench itself flags this with a TODO.

This rewrites that one method to use the prebuilt image. Run it with the grader
interpreter after every install/upgrade of swebench:

    .venv-swebench/bin/python scripts/patch_swebench_modal.py
"""

import shutil
import sys
from pathlib import Path

MARKER = "PATCHED (CodeSquire)"

START = """    @staticmethod
    def get_instance_image(test_spec: TestSpec) -> modal.Image:
"""

END = """            .workdir("/testbed/")
        )
"""

REPLACEMENT = '''    @staticmethod
    def get_instance_image(test_spec: TestSpec) -> modal.Image:
        # PATCHED (CodeSquire): swebench >=5 ships a prebuilt evaluation image per
        # instance in the dataset's `image` field, and TestSpec no longer carries
        # setup_env_script / install_repo_script. Use the prebuilt image rather
        # than rebuilding the conda environment from scratch.
        return modal.Image.from_registry(test_spec.image).workdir("/testbed/")
'''


def main() -> int:
    try:
        from swebench.harness.modal_eval import run_evaluation_modal
    except ImportError as error:
        print(f"Cannot import swebench's Modal backend: {error}", file=sys.stderr)
        return 1

    target = Path(run_evaluation_modal.__file__)
    source = target.read_text()

    if MARKER in source:
        print(f"Already patched: {target}")
        return 0

    start = source.find(START)
    if start < 0:
        print(f"Could not locate get_instance_image in {target}.", file=sys.stderr)
        print("swebench may have fixed or restructured this; re-check before patching.",
              file=sys.stderr)
        return 1

    end = source.find(END, start)
    if end < 0:
        print(f"Could not locate the end of get_instance_image in {target}.", file=sys.stderr)
        return 1
    end += len(END)

    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copy2(target, backup)

    target.write_text(source[:start] + REPLACEMENT + source[end:])
    print(f"Patched {target}")
    print(f"Backup  {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
