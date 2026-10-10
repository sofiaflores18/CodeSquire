#!/usr/bin/env python3
"""Repair the Modal grading path in swebench >=5.

swebench 5.x moved to prebuilt per-instance evaluation images, published in the
dataset's `image` column, and dropped `setup_env_script` / `install_repo_script`
from TestSpec. Its Modal backend was not migrated and still tries to rebuild the
environment from scratch, so `ModalSandboxRuntime.get_instance_image` raises
AttributeError before any sandbox starts. swebench itself flags this with a TODO.

Its Modal backend is also written against the pre-1.0 modal SDK, so
`Sandbox.open` (removed in modal 1.x in favour of `Sandbox.filesystem`) raises
AttributeError as soon as a sandbox is created.

This applies three repairs:
  1. get_instance_image  -> use the dataset's prebuilt image
  2. write_file          -> Sandbox.filesystem.write_text (modal >=1.0)
  3. the cgroup cpu.shares "hack for pylint" -> best-effort, never fatal

Run it with the grader interpreter after every install/upgrade of swebench:

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

    source = source[:start] + REPLACEMENT + source[end:]

    # modal >=1.0 removed Sandbox.open in favour of the filesystem namespace.
    # Note the reversed argument order: write_text(data, remote_path).
    old_write = """    def write_file(self, file_path: str, content: str):
        self.sandbox.open(file_path, "w").write(content)
"""
    new_write = """    def write_file(self, file_path: str, content: str):
        # PATCHED (CodeSquire): Sandbox.open was removed in modal >=1.0.
        self.sandbox.filesystem.write_text(content, file_path)
"""
    if source.count(old_write) != 1:
        print(f"Could not locate write_file in {target}.", file=sys.stderr)
        return 1
    source = source.replace(old_write, new_write)

    # This tunes cpu.shares for pylint instances only. /sys/fs/cgroup is often
    # read-only in a sandbox, and a failure here must not abort the evaluation.
    old_hack = """        # Hack for pylint
        self.write_file("/sys/fs/cgroup/cpu/cpu.shares", "2048")
"""
    new_hack = """        # Hack for pylint. PATCHED (CodeSquire): best-effort only -- /sys/fs/cgroup
        # may be read-only, and this tuning is not required for correctness.
        try:
            self.write_file("/sys/fs/cgroup/cpu/cpu.shares", "2048")
        except Exception as error:
            print(f"Skipping cpu.shares tuning: {error}")
"""
    if source.count(old_hack) != 1:
        print(f"Could not locate the cpu.shares hack in {target}.", file=sys.stderr)
        return 1
    source = source.replace(old_hack, new_hack)

    target.write_text(source)
    print(f"Patched {target}")
    print(f"Backup  {backup}")
    print("Applied: get_instance_image, write_file, cpu.shares hack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
