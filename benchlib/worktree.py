import argparse
import logging
import multiprocessing
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_project_path_from_worktree(worktree_path: pathlib.Path) -> pathlib.Path:
    git_file = worktree_path / ".git"
    if not git_file.is_file():
        raise ValueError(f".git file not found in worktree: {git_file}")
    with open(git_file, "r", encoding="utf-8") as fp:
        first = fp.readline().strip()
    m = re.match(r"gitdir: (.*?)/\.git/worktrees/[^/]+/?$", first)
    if not m:
        raise ValueError(f"Cannot derive project path from {git_file!s}: {first!r}")
    return pathlib.Path(m.group(1)).resolve()


def _git_generic(root, *git_args: str):
    try:
        return subprocess.run(
            ["git", *git_args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd} in {root}\n{e.stderr}", file=sys.stderr)
        raise


def cleanup_worktree(project_path: pathlib.Path, worktree_path: pathlib.Path, delete_agent_log: bool = True):
    if worktree_path.exists():
        logger.info(f"Cleaning up worktree {worktree_path}")
    try:
        _git_generic(project_path, "worktree", "remove", "--force", "--force", str(worktree_path))
    except subprocess.CalledProcessError:
        if worktree_path.exists():
            logger.error(f"git worktree remove failed, falling back to rmtree for {worktree_path}")
            shutil.rmtree(worktree_path, ignore_errors=True)

    agent_log_path = worktree_path.parent / f"{worktree_path.name}-agent.txt"
    harness_log_path = worktree_path.parent / f"{worktree_path.name}-harness-tests.txt"

    for log_path, label in [(agent_log_path, "agent log"), (harness_log_path, "harness log")]:
        if delete_agent_log and log_path.exists():
            try:
                log_path.unlink()
            except OSError as e:
                logger.error(f"Error deleting {label} {log_path}: {e}")
        elif log_path.exists():
            logger.info(f"{label.capitalize()} preserved for debugging: {log_path}")


def archive_worktree(project_path: pathlib.Path, worktree_path: pathlib.Path):
    project_path = pathlib.Path(project_path)
    worktree_path = pathlib.Path(worktree_path)
    archive_success = False

    try:
        if not worktree_path.is_dir():
            return

        run_output = worktree_path / "run-output.txt"
        llm_history_dir = worktree_path / ".brokk" / "llm-history"

        commits_exist = False
        git_marker = worktree_path / ".git"
        if git_marker.exists():
            try:
                _git_generic(worktree_path, "log", "-1", "--grep=^Agent work$", "HEAD")
                _git_generic(
                    worktree_path, "log", "-1", "--grep=^BrokkBench: extract test files$", "HEAD~1"
                )
                commits_exist = True
            except subprocess.CalledProcessError:
                pass

        missing = []
        if not run_output.exists():
            missing.append("run-output.txt")
        if not llm_history_dir.is_dir():
            missing.append(".brokk/llm-history/")
        if not commits_exist:
            missing.append("git commits (Agent work / extract test files)")

        if missing:
            missing_str = ", ".join(missing)
            logger.warning(f"Required files [{missing_str}] missing in {worktree_path}, skipping archive. Agent log preserved for debugging.")
            return

        zip_path = worktree_path.parent / f"{worktree_path.name}.zip"
        logger.info(f"Archiving worktree {worktree_path} to {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(run_output, arcname="run-output.txt")

            harness_log_path = worktree_path.parent / f"{worktree_path.name}-harness-tests.txt"
            if harness_log_path.exists():
                zf.write(harness_log_path, arcname="harness-tests.txt")

            selected_tests_brokk = worktree_path / ".brokk" / "selected-tests.txt"
            if selected_tests_brokk.exists():
                zf.write(selected_tests_brokk, arcname=".brokk/selected-tests.txt")

            for root, _, files in os.walk(llm_history_dir):
                for file in files:
                    file_path = pathlib.Path(root) / file
                    arcname = file_path.relative_to(worktree_path)
                    zf.write(file_path, arcname=str(arcname))

            tests_diff = _git_generic(worktree_path, "diff", "HEAD~1^", "HEAD~1").stdout
            zf.writestr("01-tests.diff", tests_diff)

            agent_diff = _git_generic(worktree_path, "diff", "HEAD~1", "HEAD").stdout
            zf.writestr("02-agent.diff", agent_diff)

        logger.info(f"Successfully created archive {zip_path}")
        archive_success = True

    finally:
        cleanup_worktree(project_path, worktree_path, delete_agent_log=archive_success)


def _archive_worker(worktree_path_str: str):
    worktree_path = pathlib.Path(worktree_path_str)
    try:
        project_path = _get_project_path_from_worktree(worktree_path)
        archive_worktree(project_path, worktree_path)
    except (ValueError, subprocess.CalledProcessError) as e:
        logger.error(f"Error processing {worktree_path}: {e}")
        logger.warning(f"Attempting to clean up directory {worktree_path}")
        if worktree_path.is_dir():
            shutil.rmtree(worktree_path, ignore_errors=True)
        agent_log_path = worktree_path.parent / f"{worktree_path.name}-agent.txt"
        harness_log_path = worktree_path.parent / f"{worktree_path.name}-harness-tests.txt"
        for log_path in [agent_log_path, harness_log_path]:
            if log_path.exists():
                try:
                    log_path.unlink()
                except OSError as exc:
                    print(f"Error deleting log {log_path}: {exc}", file=sys.stderr)


def archive_main():
    logging.basicConfig()
    parser = argparse.ArgumentParser(description="Archive one or more Brokk worktrees.")
    parser.add_argument(
        "targets",
        nargs="*",
        help="Path(s) to worktrees to archive. If none are provided, read from stdin.",
    )
    args = parser.parse_args()

    targets = args.targets
    if not targets:
        targets = [line.strip() for line in sys.stdin if line.strip()]

    if not targets:
        parser.print_help(sys.stderr)
        sys.exit(1)

    invalid_targets = [t for t in targets if not pathlib.Path(t).is_dir()]
    if invalid_targets:
        for target in invalid_targets:
            logger.error(f"Error: target path does not exist or is not a directory: {target}")
        sys.exit("Aborting due to invalid targets.")

    with multiprocessing.Pool() as pool:
        list(
            tqdm(
                pool.imap_unordered(_archive_worker, targets),
                total=len(targets),
                desc="Archiving worktrees",
            )
        )
