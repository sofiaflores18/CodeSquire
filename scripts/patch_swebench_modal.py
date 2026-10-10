#!/usr/bin/env python3
"""Repair the Modal grading path in swebench >=5.

Two independent things are broken:

1. swebench 5.x moved to prebuilt per-instance evaluation images, published in
   the dataset's `image` column, and dropped `setup_env_script` /
   `install_repo_script` from TestSpec. Its Modal backend was not migrated and
   still tries to rebuild the environment from scratch, so
   `ModalSandboxRuntime.get_instance_image` raises AttributeError before any
   sandbox starts. swebench itself flags this with a TODO.
2. The same backend targets the pre-1.0 modal SDK. `Sandbox.open` was removed in
   modal 1.x in favour of the `Sandbox.filesystem` namespace, so `write_file`
   raises AttributeError as soon as a sandbox is created.

Three repairs are applied:
  1. get_instance_image -> use the dataset's prebuilt image
  2. write_file         -> Sandbox.filesystem.write_text (modal >=1.0)
  3. the cgroup cpu.shares "hack for pylint" -> best-effort, never fatal

Each repair is checked and applied independently, so running an updated copy of
this script over a partially patched install correctly tops it up.

Run it with the grader interpreter after every install/upgrade of swebench:

    .venv-swebench/bin/python scripts/patch_swebench_modal.py
"""

import shutil
import sys
from pathlib import Path

# ---- repair 1: prebuilt instance image ------------------------------------
IMAGE_START = """    @staticmethod
    def get_instance_image(test_spec: TestSpec) -> modal.Image:
"""

IMAGE_END = """            .workdir("/testbed/")
        )
"""

IMAGE_NEW = '''    @staticmethod
    def get_instance_image(test_spec: TestSpec) -> modal.Image:
        # PATCHED (CodeSquire): swebench >=5 ships a prebuilt evaluation image per
        # instance in the dataset's `image` field, and TestSpec no longer carries
        # setup_env_script / install_repo_script. Use the prebuilt image rather
        # than rebuilding the conda environment from scratch.
        return modal.Image.from_registry(test_spec.image).workdir("/testbed/")
'''

IMAGE_DONE = "return modal.Image.from_registry(test_spec.image)"

# ---- repair 2: modal >=1.0 filesystem API --------------------------------
WRITE_OLD = '''    def write_file(self, file_path: str, content: str):
        self.sandbox.open(file_path, "w").write(content)
'''

WRITE_NEW = '''    def write_file(self, file_path: str, content: str):
        # PATCHED (CodeSquire): Sandbox.open was removed in modal >=1.0; the
        # filesystem namespace takes (data, remote_path) in that order.
        self.sandbox.filesystem.write_text(content, file_path)
'''

WRITE_DONE = "self.sandbox.filesystem.write_text(content, file_path)"

# ---- repair 3: cpu.shares tuning must not be fatal -----------------------
HACK_OLD = '''        # Hack for pylint
        self.write_file("/sys/fs/cgroup/cpu/cpu.shares", "2048")
'''

HACK_NEW = '''        # Hack for pylint. PATCHED (CodeSquire): best-effort only -- /sys/fs/cgroup
        # may be read-only, and this tuning is not required for correctness.
        try:
            self.write_file("/sys/fs/cgroup/cpu/cpu.shares", "2048")
        except Exception as error:
            print(f"Skipping cpu.shares tuning: {error}")
'''

HACK_DONE = "Skipping cpu.shares tuning"


def apply_span(source: str, name: str, start: str, end: str, new: str, done: str):
    """Replace the whole start..end span. Returns (source, status)."""
    if done in source:
        return source, "already applied"
    begin = source.find(start)
    if begin < 0:
        raise RuntimeError(f"could not locate {name}")
    stop = source.find(end, begin)
    if stop < 0:
        raise RuntimeError(f"could not locate the end of {name}")
    return source[:begin] + new + source[stop + len(end):], "applied"


def apply_exact(source: str, name: str, old: str, new: str, done: str):
    """Replace one exact block. Returns (source, status)."""
    if done in source:
        return source, "already applied"
    if source.count(old) != 1:
        raise RuntimeError(f"could not locate {name} (found {source.count(old)} matches)")
    return source.replace(old, new), "applied"


def main() -> int:
    try:
        from swebench.harness.modal_eval import run_evaluation_modal
    except ImportError as error:
        print(f"Cannot import swebench's Modal backend: {error}", file=sys.stderr)
        return 1

    target = Path(run_evaluation_modal.__file__)
    original = target.read_text()
    source = original
    results = {}

    try:
        source, results["get_instance_image"] = apply_span(
            source, "get_instance_image", IMAGE_START, IMAGE_END, IMAGE_NEW, IMAGE_DONE
        )
        source, results["write_file"] = apply_exact(
            source, "write_file", WRITE_OLD, WRITE_NEW, WRITE_DONE
        )
        source, results["cpu.shares hack"] = apply_exact(
            source, "the cpu.shares hack", HACK_OLD, HACK_NEW, HACK_DONE
        )
    except RuntimeError as error:
        print(f"{target}: {error}", file=sys.stderr)
        print("swebench may have fixed or restructured this; re-check before patching.",
              file=sys.stderr)
        return 1

    if source == original:
        print(f"Up to date, no changes needed: {target}")
    else:
        backup = target.with_suffix(".py.orig")
        if not backup.exists():
            shutil.copy2(target, backup)
            print(f"Backup  {backup}")
        target.write_text(source)
        print(f"Patched {target}")

    for name, status in results.items():
        print(f"  {status:<15} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
