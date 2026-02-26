from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import random
import sys
from collections import defaultdict
from collections import deque
from collections.abc import Callable, Iterable
import tempfile
import re
import threading

from dotenv import load_dotenv

import tasks

import benchlib.cli
from benchlib.cli import set_cli_bin, validate_api_key, validate_cache_dirs, validate_models
from benchlib.git import git_run as _git
from benchlib.run import run_pipelined_tasks, RunOutcome, RunResult, Task
from dataset_config import get_dataset_config, build_job_env
from scan_commits import has_test_word


ExitFn = Callable[[int], None]
TaskScheduledCallback = Callable[[Task], None]
TaskStartedCallback = Callable[[Task, pathlib.Path, int], None]
TaskCompletedCallback = Callable[[Task, RunResult], None]
TaskHistorySeededCallback = Callable[[dict[str, int], dict[str, int]], None]
TaskKey = tuple[str, str, str, int]
PrioritySpec = tuple[str, tuple[str, ...]]

_ZERO_REMAINING_STOP_REASON = "MERCY_RULE"


def task_key(task: Task) -> TaskKey:
    return (task.project, task.revision, task.model, task.run_number)


class TaskWorktreeMap:
    """Thread-safe mapping from task identity to active worktree path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths: dict[TaskKey, pathlib.Path] = {}

    def mark_scheduled(self, task: Task) -> TaskKey:
        key = task_key(task)
        with self._lock:
            self._paths.pop(key, None)
        return key

    def mark_started(self, task: Task, worktree_path: pathlib.Path) -> TaskKey:
        key = task_key(task)
        with self._lock:
            self._paths[key] = worktree_path
        return key

    def mark_completed(self, task: Task) -> TaskKey:
        key = task_key(task)
        with self._lock:
            self._paths.pop(key, None)
        return key

    def get(self, task: Task) -> pathlib.Path | None:
        return self.get_by_key(task_key(task))

    def get_by_key(self, key: TaskKey) -> pathlib.Path | None:
        with self._lock:
            return self._paths.get(key)


class _BprOutput:
    def __init__(
        self,
        *,
        stdout: Callable[[str], None] | None = None,
        stderr: Callable[[str], None] | None = None,
    ) -> None:
        self._stdout = stdout or (lambda message: print(message, file=sys.stdout))
        self._stderr = stderr or (lambda message: print(message, file=sys.stderr))

    def stdout(self, message: str) -> None:
        self._stdout(message)

    def stderr(self, message: str) -> None:
        self._stderr(message)


def _resolve_output(output: _BprOutput | None = None) -> _BprOutput:
    return output if output is not None else _default_output()


def _default_output() -> _BprOutput:
    return _BprOutput()


def _write_exit(code: int, exit_fn: ExitFn = sys.exit) -> None:
    exit_fn(code)


def _read_revisions_from_lines(lines: Iterable[str]) -> list[str]:
    revisions: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            revisions.append(stripped.split()[0])
    return revisions


def _parse_priority_spec(value: str) -> PrioritySpec:
    if "=" not in value:
        if value in {"hardest", "easiest", "dataset", "model", "random"}:
            return value, ()
        raise argparse.ArgumentTypeError(
            "Invalid --priority value. Valid formats: hardest, easiest, dataset, model, random, "
            "dataset=<dataset1,dataset2>, model=<model1,model2>",
        )

    mode, raw_values = value.split("=", 1)
    if mode not in {"dataset", "model"}:
        raise argparse.ArgumentTypeError(
            f"Invalid --priority mode {mode!r} for explicit list mode. Use dataset=<...> or model=<...>.",
        )

    values = tuple(v.strip() for v in raw_values.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"Expected at least one value after {mode}= for --priority.")
    return mode, values


def _normalize_priority_spec(priority: str | PrioritySpec) -> PrioritySpec:
    if isinstance(priority, tuple):
        return priority
    return _parse_priority_spec(priority)


def _load_filter_revisions(
    filter_file: pathlib.Path | None,
    *,
    output: _BprOutput | None = None,
    exit_fn: ExitFn = sys.exit,
) -> set[str] | None:
    output = _resolve_output(output)
    if filter_file is None:
        return None
    if not filter_file.is_file():
        output.stderr(f"Error: filter file {filter_file} not found")
        _write_exit(1, exit_fn)
        return set()

    try:
        with open(filter_file, "r", encoding="utf-8") as fp:
            revisions: list[str] = []
            for line_num, raw_line in enumerate(fp, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    output.stderr(
                        f"Error: filter file {filter_file} is not valid JSONL (line {line_num})"
                    )
                    _write_exit(1, exit_fn)
                    return set()
                if not isinstance(payload, dict):
                    output.stderr(f"Error: filter file {filter_file} invalid row on line {line_num}")
                    _write_exit(1, exit_fn)
                    return set()
                revision = payload.get("hash")
                if not isinstance(revision, str) or not revision.strip():
                    output.stderr(
                        f"Error: filter file {filter_file} missing revision hash on line {line_num}"
                    )
                    _write_exit(1, exit_fn)
                    return set()
                revisions.append(revision.strip())
    except OSError as exc:
        output.stderr(f"Error: unable to read filter file {filter_file}: {exc}")
        _write_exit(1, exit_fn)
        return set()

    return set(revisions)


def _apply_revision_filter(revisions: list[str], filter_revisions: set[str] | None) -> list[str]:
    if filter_revisions is None:
        return revisions
    return [rev for rev in revisions if rev in filter_revisions]


def _load_props(tasks_dir: pathlib.Path, revision: str) -> dict[str, str]:
    props: dict[str, str] = {}
    props_path = tasks_dir / f"{revision}.properties"
    if not props_path.exists():
        return props
    with open(props_path, "r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            props[k.strip()] = v.strip()
    return props


def _build_cli_args_for_job(
    project_path: pathlib.Path,
    tasks_dir: pathlib.Path,
    revision: str,
    model: str,
    planmodel: str | None,
    base_cli_tokens: list[str],
    exclude_patterns: list[str],
    mode: str,
) -> tuple[list[str], list[str], list[str]]:
    """
    Compose per-job CLI args:
      - --code=@<tasks_dir>/<revision>.txt (default)
      - --infer-context=@<tasks_dir>/<revision>.txt (when mode == 'infer-context')
      - --edit=... for edit files (excluding patterns)
      - --read=... for test files (+ extra_tests, - extra_edits)
      - --codemodel=<model>
      - --planmodel=<model>
      - plus any user-supplied base_cli_tokens (excluding --project/--worktree)
    Returns (args, edit_files, test_files)
    """
    # Determine edited and test files from git diff
    diff_lines = _git(project_path, "diff", "--name-status", "--no-renames", f"{revision}^", revision).splitlines()
    edit_files: list[str] = []
    test_files: list[str] = []
    for line in diff_lines:
        if not line.strip():
            continue
        status, path = line.split("\t", 1)
        path = path.strip()
        if has_test_word(path):
            if status != "D":
                test_files.append(path)
        else:
            edit_files.append(path)

    # Load properties for extra_tests. Exclude patterns from properties
    # are merged into the global exclude list earlier in main.
    props = _load_props(tasks_dir, revision)

    # Merge extra tests from properties (if any)
    extra_tests_raw = props.get("extra_tests", "")
    extra_tests = [p.strip() for p in extra_tests_raw.split(",") if p.strip()]
    for et in extra_tests:
        if et not in test_files:
            test_files.append(et)

    # Extra edits from shared properties (populated by tasktune.py)
    extra_edits: list[str] = [p.strip() for p in props.get("files", "").split(",") if p.strip()]
    for ee in extra_edits:
        if ee not in edit_files:
            edit_files.append(ee)
    test_files = [tf for tf in test_files if tf not in extra_edits]

    # Use exclude_patterns as provided by caller; properties-based excludes should
    # have already been merged into that list by the caller (main).
    combined_exclude_patterns = exclude_patterns

    # Build args
    args: list[str] = []

    tasks_dir_posix = tasks_dir.as_posix()
    if mode == "infer-context":
        args.append(f"--infer-context=@{tasks_dir_posix}/{revision}.txt")
    else:
        args.append(f"--code=@{tasks_dir_posix}/{revision}.txt")

    # --edit (filter excludes, but always include extra_edits)
    filtered_edit_files: list[str] = []
    for f in edit_files:
        # Skip excluded patterns, unless the file is from extra_edits
        if f not in extra_edits and any(pathlib.Path(f).match(pat) for pat in combined_exclude_patterns):
            continue
        filtered_edit_files.append(f)
        args.append(f"--edit={f}")

    # --read (only from our computed list)
    for f in test_files:
        if f not in extra_tests and any(pathlib.Path(f).match(pat) for pat in combined_exclude_patterns):
            continue
        args.append(f"--read={f}")

    # --model
    args.append(f"--codemodel={model}")
    args.append(f"--planmodel={planmodel if planmodel else model}")

    # finally merge user supplied tokens (filter out --project/--worktree if present)
    i = 0
    n = len(base_cli_tokens)
    while i < n:
        tok = base_cli_tokens[i]
        if tok in ("--project", "--worktree"):
            i += 2 if (i + 1 < n and not base_cli_tokens[i + 1].startswith("--")) else 1
            continue
        if tok.startswith("--project=") or tok.startswith("--worktree="):
            i += 1
            continue
        args.append(tok)
        i += 1

    return args, filtered_edit_files, test_files


def find_missing_jobs(
    project_name: str,
    results_root: pathlib.Path,
    revisions: list[str],
    models: list[str],
    runs: int,
) -> list[tuple[str, str, int]]:
    """
    Determine which (revision, model, run_number) combinations are missing result files.
    Returns a list of (revision, model, run_number) tuples.
    """
    missing_jobs: list[tuple[str, str, int]] = []

    for run_number in range(1, runs + 1):
        results_dir = results_root / f"{project_name}{run_number}"

        for rev in revisions:
            for model in models:
                res_path = results_dir / f"{model}-{rev}.json"
                if not res_path.exists():
                    missing_jobs.append((rev, model, run_number))

    return missing_jobs


def _results_json_path(
    results_root: pathlib.Path,
    project_name: str,
    run_number: int,
    model: str,
    revision: str,
) -> pathlib.Path:
    return tasks.build_result_path(results_root, project_name, run_number, model, revision)


def _read_run_outcome_from_results_file(path: pathlib.Path) -> RunOutcome | None:
    if not path.exists():
        return None
    stop_reason = tasks.read_result_stop_reason(path)
    if stop_reason is None:
        return None
    token = tasks.outcome_token(stop_reason)
    if token == tasks.RUN_OUTCOME_SUCCESS:
        return RunOutcome.SUCCESS
    if token == tasks.RUN_OUTCOME_TESTS_FAILED:
        return RunOutcome.TESTS_FAILED
    if token == tasks.RUN_OUTCOME_AGENT_FAILED:
        return RunOutcome.AGENT_FAILED
    return None


def _read_stop_reason_from_results_file(path: pathlib.Path) -> str | None:
    return tasks.read_result_stop_reason(path)


def _project_max_run(results_root: pathlib.Path, project_name: str) -> int | None:
    max_run: int | None = None
    if not results_root.exists():
        return None

    for entry in results_root.iterdir():
        if not entry.is_dir():
            continue
        discovered_project, run_number = tasks.parse_project_and_run(entry.name)
        if discovered_project != project_name or run_number is None:
            continue
        if max_run is None or run_number > max_run:
            max_run = run_number

    return max_run


def _build_zero_remaining_plan_for_project(
    *,
    project_name: str,
    revisions: list[str],
    models: list[str],
    results_root: pathlib.Path,
) -> tuple[int | None, dict[str, int], dict[str, int], list[pathlib.Path]]:
    max_run = _project_max_run(results_root, project_name)
    existing_real = {model: 0 for model in models}
    fake_to_write = {model: 0 for model in models}
    write_paths: list[pathlib.Path] = []

    if max_run is None:
        return None, existing_real, fake_to_write, write_paths

    for revision in revisions:
        for model in models:
            previous_outcome: RunOutcome | None = None

            for run_number in range(1, max_run + 1):
                if not _should_run_in_run("failed", run_number, previous_outcome):
                    break

                result_path = _results_json_path(results_root, project_name, run_number, model, revision)

                if not result_path.exists():
                    fake_to_write[model] += 1
                    write_paths.append(result_path)
                    break

                stop_reason = _read_stop_reason_from_results_file(result_path)
                if stop_reason != _ZERO_REMAINING_STOP_REASON:
                    existing_real[model] += 1

                previous_outcome = _read_run_outcome_from_results_file(result_path)
                if previous_outcome is None:
                    previous_outcome = RunOutcome.AGENT_FAILED

    return max_run, existing_real, fake_to_write, write_paths


def _write_zero_remaining_placeholder(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump({"stopReason": _ZERO_REMAINING_STOP_REASON}, fp, separators=(",", ":"), ensure_ascii=False)
        fp.write("\n")


def _confirm_zero_remaining_write(
    files_to_write: int,
    *,
    output: _BprOutput | None = None,
    input_fn: Callable[[str], str] = input,
) -> bool:
    output = _resolve_output(output)
    prompt = f"Write {files_to_write} fake MERCY_RULE results? [y/N]: "
    try:
        answer = input_fn(prompt)
    except EOFError:
        output.stderr("No confirmation received (EOF); aborting write.")
        return False
    return answer.strip().lower() in {"y", "yes"}


def _collect_zero_remaining_single_project_entries(
    args: argparse.Namespace,
    filter_revisions: set[str] | None,
    *,
    output: _BprOutput | None = None,
    exit_fn: ExitFn = sys.exit,
) -> list[dict] | None:
    output = _resolve_output(output)
    project_path = pathlib.Path(args.project).resolve()
    tasks_dir = pathlib.Path(args.tasksdir).resolve()

    revisions = _read_revisions_from_lines(sys.stdin)
    revisions = _apply_revision_filter(revisions, filter_revisions)

    existing_revisions = [revision for revision in revisions if (tasks_dir / f"{revision}.txt").exists()]
    if len(revisions) != len(existing_revisions):
        output.stderr(
            f"Warning: {len(revisions)} revisions requested, but only {len(existing_revisions)} tasks found on disk",
        )
    revisions = existing_revisions

    if not revisions:
        output.stderr("No valid revisions with task files provided on stdin.")
        _write_exit(3, exit_fn)
        return None

    valid_revisions = _validate_revisions(project_path, revisions, output=output)
    if len(valid_revisions) != len(revisions):
        dropped = set(revisions) - set(valid_revisions)
        output.stderr("Validation failed for revisions: " + ", ".join(dropped))
        _write_exit(4, exit_fn)
        return None

    return [{"project_path": project_path, "revisions": valid_revisions}]


def _collect_zero_remaining_multi_project_entries(
    args: argparse.Namespace,
    filter_revisions: set[str] | None,
    *,
    output: _BprOutput | None = None,
    exit_fn: ExitFn = sys.exit,
) -> list[dict] | None:
    output = _resolve_output(output)
    from dataset_config import DEFAULT_DATASETS

    if args.projects:
        dataset_names = [name.strip() for name in args.projects.split(",") if name.strip()]
    else:
        dataset_names = DEFAULT_DATASETS

    tasks_dir = pathlib.Path(args.tasksdir).resolve()
    commits_dir = args.commits_dir

    project_entries: list[dict] = []
    for dataset_name in dataset_names:
        try:
            config = get_dataset_config(dataset_name, commits_dir)
        except ValueError as exc:
            output.stderr(f"Error: {exc}")
            _write_exit(1, exit_fn)
            return None

        project_path = pathlib.Path(config.project_path).resolve()
        if not project_path.is_dir():
            output.stderr(f"Error: project directory {project_path} does not exist for dataset '{dataset_name}'")
            _write_exit(1, exit_fn)
            return None

        commits_path = pathlib.Path(config.commits_file)
        if not commits_path.is_file():
            output.stderr(f"Error: commits file {commits_path} not found for dataset '{dataset_name}'")
            _write_exit(1, exit_fn)
            return None

        revisions = _read_revisions_from_file(config.commits_file)
        revisions = _apply_revision_filter(revisions, filter_revisions)

        existing_revisions = [revision for revision in revisions if (tasks_dir / f"{revision}.txt").exists()]
        if len(revisions) != len(existing_revisions):
            output.stderr(
                f"Warning ({dataset_name}): {len(revisions)} revisions requested, "
                f"but only {len(existing_revisions)} tasks found on disk",
            )
        revisions = existing_revisions

        if not revisions:
            output.stderr(f"Warning ({dataset_name}): no valid revisions, skipping")
            continue

        revisions = _validate_revisions(project_path, revisions, output=output)
        if not revisions:
            output.stderr(f"Warning ({dataset_name}): all revisions failed validation, skipping")
            continue

        project_entries.append({"project_path": project_path, "revisions": revisions})

    return project_entries


def _run_zero_remaining(
    args: argparse.Namespace,
    models: list[str],
    results_root: pathlib.Path,
    filter_revisions: set[str] | None,
    *,
    output: _BprOutput | None = None,
    exit_fn: ExitFn = sys.exit,
    input_fn: Callable[[str], str] = input,
) -> int:
    output = _resolve_output(output)

    if args.project:
        project_entries = _collect_zero_remaining_single_project_entries(
            args,
            filter_revisions,
            output=output,
            exit_fn=exit_fn,
        )
    else:
        project_entries = _collect_zero_remaining_multi_project_entries(
            args,
            filter_revisions,
            output=output,
            exit_fn=exit_fn,
        )

    if project_entries is None:
        return 0
    if not project_entries:
        output.stderr("No projects with valid revisions available for --zero-remaining.")
        _write_exit(0, exit_fn)
        return 0

    existing_real_totals = {model: 0 for model in models}
    fake_to_write_totals = {model: 0 for model in models}
    paths_to_write: list[pathlib.Path] = []
    projects_with_runs = 0

    for entry in project_entries:
        project_path: pathlib.Path = entry["project_path"]
        revisions: list[str] = entry["revisions"]
        max_run, existing_real, fake_to_write, write_paths = _build_zero_remaining_plan_for_project(
            project_name=project_path.name,
            revisions=revisions,
            models=models,
            results_root=results_root,
        )
        if max_run is None:
            output.stderr(
                f"Project {project_path.name}: no existing run directories in {results_root}, skipping",
            )
            continue

        projects_with_runs += 1
        output.stderr(f"Project {project_path.name}: pR={max_run}, revisions={len(revisions)}")
        for model in models:
            existing_real_totals[model] += existing_real.get(model, 0)
            fake_to_write_totals[model] += fake_to_write.get(model, 0)
        paths_to_write.extend(write_paths)

    if projects_with_runs == 0:
        output.stderr("No projects have existing run directories; nothing to fill.")
        _write_exit(0, exit_fn)
        return 0

    output.stderr("--zero-remaining summary--")
    for model in models:
        output.stderr(
            f"  {model}: existing(real)={existing_real_totals.get(model, 0)}, "
            f"fake(to_write)={fake_to_write_totals.get(model, 0)}",
        )

    total_to_write = len(paths_to_write)
    output.stderr(f"Total fake files to write: {total_to_write}")

    if total_to_write == 0:
        output.stderr("No missing task+run pairs found.")
        _write_exit(0, exit_fn)
        return 0

    if not _confirm_zero_remaining_write(total_to_write, output=output, input_fn=input_fn):
        output.stderr("Aborted; no files written.")
        _write_exit(0, exit_fn)
        return 0

    for path in paths_to_write:
        _write_zero_remaining_placeholder(path)
    output.stderr(f"Wrote {total_to_write} MERCY_RULE placeholder files.")
    _write_exit(0, exit_fn)
    return 0


def _should_run_in_run(
    rerun_mode: str,
    run_number: int,
    previous_outcome: RunOutcome | None,
) -> bool:
    if previous_outcome == RunOutcome.SUCCESS:
        previous = tasks.RUN_OUTCOME_SUCCESS
    elif previous_outcome == RunOutcome.TESTS_FAILED:
        previous = tasks.RUN_OUTCOME_TESTS_FAILED
    elif previous_outcome == RunOutcome.AGENT_FAILED:
        previous = tasks.RUN_OUTCOME_AGENT_FAILED
    else:
        previous = None
    return tasks.should_run_in_rerun_mode(rerun_mode, run_number, previous)


def _report_tasks_remaining_for_run(
    tasks: list[Task],
    models: list[str],
    run_number: int,
    project_label: str | None = None,
    avg_elapsed_ms: float = 0.0,
    threads: int = 1,
    *,
    exact: bool = True,
    unknown: bool = False,
    counts_by_model_override: dict[str, int] | None = None,
    output: _BprOutput | None = None,
) -> None:
    output = _resolve_output(output)

    if project_label:
        output.stderr(f"--- Project: {project_label} ---")

    output.stderr(f"--- Run {run_number} ---")

    if unknown:
        output.stderr("??? Tasks Remaining")
        for model in sorted(models):
            output.stderr(f"  {model}: ???")
        output.stderr("")
        return

    if counts_by_model_override is None:
        counts_by_model = {m: 0 for m in models}
        for t in tasks:
            if t.model in counts_by_model:
                counts_by_model[t.model] += 1
        total = len(tasks)
    else:
        counts_by_model = {m: int(counts_by_model_override.get(m, 0)) for m in models}
        total = sum(counts_by_model.values())

    total_label = str(total) if exact else f">= {total}"
    output.stderr(f"{total_label} Tasks Remaining")

    for model in sorted(models):
        c = counts_by_model.get(model, 0)
        c_label = str(c) if exact else f">= {c}"
        output.stderr(f"  {model}: {c_label}")
    output.stderr("")

    if avg_elapsed_ms > 0 and total > 0 and exact:
        effective_threads = min(threads, total)
        eta_ms = (total * avg_elapsed_ms) / effective_threads
        output.stderr(
            f"ETA: ~{_format_duration(eta_ms)} ({total} tasks, "
            f"avg {_format_duration(avg_elapsed_ms)}/task, {effective_threads} threads)",
        )


def make_commit_tests(exclude_patterns: list[str]) -> Callable[[pathlib.Path, pathlib.Path, str, dict[str, str]], None]:
    def _commit_tests(project_path_param: pathlib.Path, worktree_path: pathlib.Path, revision_param: str, env: dict[str, str]) -> None:
        diff_lines = _git(project_path_param, "diff", "--name-status", "--no-renames", f"{revision_param}^", revision_param).splitlines()
        edit_files: list[str] = []
        test_files: list[str] = []
        for line in diff_lines:
            if not line.strip():
                continue
            status, path = line.split("\t", 1)
            path = path.strip()
            if path and path != "/dev/null":
                if has_test_word(path):
                    if status != "D":
                        test_files.append(path)
                else:
                    if not any(pathlib.Path(path).match(pat) for pat in exclude_patterns):
                        edit_files.append(path)

        for tf in test_files:
            subprocess.run(
                ["git", "checkout", revision_param, "--", tf],
                cwd=worktree_path,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            subprocess.run(["git", "add", "--", tf], cwd=worktree_path, check=True, text=True)

        for ef in edit_files:
            abs_path = worktree_path / ef
            if not abs_path.exists():
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.touch()
                subprocess.run(["git", "add", "--", ef], cwd=worktree_path, check=True, text=True)

        tests_list_file = worktree_path / ".brokk" / "selected-tests.txt"
        tests_list_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tests_list_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(test_files))

        subprocess.run(
            ["git", "commit", "-m", "BrokkBench: extract test files"],
            cwd=worktree_path,
            check=True,
            text=True,
            capture_output=True,
        )
    return _commit_tests


def make_execute_tests() -> Callable[[pathlib.Path, pathlib.Path, dict[str, str], dict[str, str]], subprocess.CompletedProcess]:
    def _execute(project_path_param: pathlib.Path, worktree_path: pathlib.Path, env: dict[str, str], task_props: dict[str, str]) -> subprocess.CompletedProcess:
        log_path = worktree_path.parent / f"{worktree_path.name}-harness-tests.txt"
        with open(log_path, "wb") as log:
            strategy = task_props.get("BPR_TESTS", env.get("BPR_TESTS", "")).lower()

            test_command: str | None = None
            template: str | None = None
            selected_tests_path = worktree_path / ".brokk" / "selected-tests.txt"

            if strategy == "all":
                test_command = env.get("BRK_TESTALL_CMD")
                if test_command and ("{{#" in test_command):
                    template = test_command
                    test_command = None
            elif strategy == "some":
                env_template = env.get("BRK_TESTSOME_CMD")
                if env_template:
                    template = env_template
                else:
                    properties_file = project_path_param / ".brokk" / "project.properties"
                    build_details_json_str: str | None = None
                    if properties_file.exists():
                        with open(properties_file, "r", encoding="utf-8") as fp:
                            for line in fp:
                                stripped = line.strip()
                                if not stripped or stripped.startswith("#"):
                                    continue
                                if stripped.startswith("buildDetailsJson="):
                                    build_details_json_str = stripped.split("=", 1)[1].strip()
                                    build_details_json_str = build_details_json_str.replace(r"\:", ":")
                                    break
                    if build_details_json_str is not None:
                        try:
                            build_details = json.loads(build_details_json_str)
                            template = build_details.get("testSomeCommand")
                        except json.JSONDecodeError:
                            template = None
            else:
                raise ValueError(f"BPR_TEST is not set or invalid ({strategy}). Must be 'all' or 'some'.")

            if test_command:
                cmd = test_command
            elif template:
                test_files: list[str] = []
                if selected_tests_path.exists():
                    with open(selected_tests_path, "r", encoding="utf-8") as fp:
                        test_files = [ln.strip() for ln in fp if ln.strip()]

                is_files_based = "{{#files}}" in template
                is_fq_based = "{{#fqclasses}}" in template
                is_classes_based = "{{#classes}}" in template
                is_crates_based = "{{#crates}}" in template

                if sum(1 for b in (is_files_based, is_fq_based, is_classes_based, is_crates_based) if b) > 1:
                    raise ValueError("Template must contain at most one of #classes, #fqclasses, #files, or #crates")

                def _to_fq_class(path_str: str) -> str:
                    s = path_str.replace("\\", "/")
                    if not (s.endswith(".java") or s.endswith(".kt") or s.endswith(".groovy")):
                        return pathlib.Path(path_str).stem

                    s_noext = re.sub(r"\.(java|kt|groovy)$", "", s)

                    patterns = [
                        r"/src/test/(java|kotlin|groovy)/",
                        r"/src/[^/]+/(java|kotlin|groovy)/",
                        r"/test/(java|kotlin|groovy)/",
                        r"/(java|kotlin|groovy)/",
                    ]
                    rel = None
                    for pat in patterns:
                        m = re.search(pat, s_noext)
                        if m:
                            rel = s_noext[m.end():]
                            break
                    if rel is None:
                        rel = s_noext

                    return rel.replace("/", ".")

                def _to_crate(path_str: str) -> str:
                    parts = pathlib.Path(path_str).parts
                    for i, part in enumerate(parts):
                        if part == "crates" and i + 1 < len(parts):
                            return parts[i + 1]
                    return pathlib.Path(path_str).parts[0]

                import pystache
                if not any((is_files_based, is_fq_based, is_classes_based, is_crates_based)):
                    cmd = template
                    log.write(f"Harness test template is plain string: {template}\n".encode())
                else:
                    if is_files_based:
                        items = list(test_files)
                        key = "files"
                    elif is_fq_based:
                        items = sorted({_to_fq_class(p) for p in test_files if p.endswith((".java", ".kt", ".groovy", ".go", ".cs"))})
                        key = "fqclasses"
                    elif is_crates_based:
                        items = sorted({_to_crate(p) for p in test_files})
                        key = "crates"
                    else:
                        items = sorted({pathlib.Path(p).stem for p in test_files if p.endswith((".java", ".kt", ".groovy", ".go", ".cs"))})
                        key = "classes"

                    context = {
                        key: [
                            {"value": v, "first": i == 0, "last": i == len(items) - 1}
                            for i, v in enumerate(items)
                        ]
                    }
                    cmd = pystache.render(template, context)
                    log.write(f"Harness test template is {template} {context}\n".encode())
            else:
                raise ValueError(f"No test command available for strategy {strategy}")

            log.write(f"Running test command: {cmd}\n".encode())
            log.flush()

            origin_url = _git(worktree_path, "config", "--get", "remote.origin.url")
            last_part = origin_url.split("/")[-1].split(":")[-1]
            origin_repo_name = re.sub(r"\.git$", "", last_part)

            no_concurrent_builds = env.get("BRK_NO_CONCURRENT_BUILDS", "").lower() == "true"

            def _run_verification() -> subprocess.CompletedProcess:
                return subprocess.run(
                    cmd,
                    shell=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=worktree_path,
                    env=env,
                )

            if not no_concurrent_builds:
                log.write("(concurrent builds allowed)".encode())
                return _run_verification()
            else:
                lock_dir = pathlib.Path(tempfile.gettempdir()) / "brokk"
                lock_dir.mkdir(parents=True, exist_ok=True)
                lock_file = lock_dir / f"{origin_repo_name}.lock"
                lock_fp = None
                try:
                    import fcntl
                    lock_fp = open(lock_file, "w")
                    lock_fp.seek(0)
                    fcntl.lockf(lock_fp, fcntl.LOCK_EX)
                    log.write(f"Acquired build lock {lock_file}\n".encode())
                    log.flush()
                    return _run_verification()
                finally:
                    if lock_fp is not None:
                        try:
                            lock_fp.close()
                        except Exception:
                            pass
    return _execute


def _read_revisions_from_file(commits_file: str) -> list[str]:
    with open(commits_file, "r", encoding="utf-8") as fp:
        return _read_revisions_from_lines(fp)


def _validate_revisions(
    project_path: pathlib.Path,
    revisions: list[str],
    *,
    output: _BprOutput | None = None,
) -> list[str]:
    output = _resolve_output(output)
    valid: list[str] = []
    for rev in revisions:
        try:
            subprocess.run(
                ["git", "-C", str(project_path), "rev-parse", "--verify", f"{rev}^{{commit}}"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            output.stderr(f"Warning: revision {rev} not found in {project_path}")
            continue

        parent_line = subprocess.check_output(
            ["git", "-C", str(project_path), "rev-list", "--parents", "-n", "1", rev],
            text=True,
        ).strip()
        parent_count = len(parent_line.split()) - 1
        if parent_count != 1:
            output.stderr(f"Warning: revision {rev} is a merge commit, skipping")
            continue
        valid.append(rev)
    return valid


def _compute_avg_llm_millis(results_root: pathlib.Path, project_name: str,
                            revisions: list[str], runs: int) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for run_number in range(1, runs + 1):
        results_dir = results_root / f"{project_name}{run_number}"
        if not results_dir.is_dir():
            continue
        for json_file in results_dir.glob("*.json"):
            file_data = tasks.read_json_object(json_file)
            if file_data is None:
                continue
            payload = tasks.extract_result_payload(file_data)
            metadata = tasks.parse_result_path(json_file)
            if metadata is None:
                continue
            if metadata.revision not in revisions:
                continue
            llm_val = payload.get("llmMillis")
            if isinstance(llm_val, (int, float)):
                totals[metadata.revision].append(float(llm_val))

    avg: dict[str, float] = {}
    for rev in revisions:
        values = totals.get(rev)
        avg[rev] = (sum(values) / len(values)) if values else 0.0
    return avg


def _compute_avg_elapsed_millis(results_root: pathlib.Path, project_name: str,
                                runs: int) -> float:
    """Return the grand average elapsedMillis across all completed results for a project."""
    values: list[float] = []
    for run_number in range(1, runs + 1):
        results_dir = results_root / f"{project_name}{run_number}"
        if not results_dir.is_dir():
            continue
        for json_file in results_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except (json.JSONDecodeError, OSError):
                continue
            elapsed = data.get("elapsedMillis")
            if isinstance(elapsed, (int, float)):
                values.append(float(elapsed))
    return (sum(values) / len(values)) if values else 0.0


def _task_priority_key(
    priority: str | PrioritySpec,
    *,
    llm_scores: dict[tuple[str, str], float] | None = None,
) -> Callable[[Task, deque[Task]], tuple]:
    mode, prioritized = _normalize_priority_spec(priority)
    scores = llm_scores or {}
    prioritized_index = {name: index for index, name in enumerate(prioritized)}
    random_scores: dict[TaskKey, float] = {}

    if mode == "hardest":
        def _key(task: Task, _queue: deque[Task]) -> tuple:
            return (
                -scores.get((task.project, task.revision), 0.0),
                task.project,
                task.revision,
                task.model,
                task.run_number,
            )
        return _key

    if mode == "easiest":
        def _key(task: Task, _queue: deque[Task]) -> tuple:
            return (
                scores.get((task.project, task.revision), 0.0),
                task.project,
                task.revision,
                task.model,
                task.run_number,
            )
        return _key

    if mode == "dataset":
        if prioritized:
            def _key(task: Task, queue: deque[Task]) -> tuple:
                rank = prioritized_index.get(task.project)
                if rank is None:
                    return (
                        1,
                        task.project,
                        task.revision,
                        task.model,
                        task.run_number,
                    )
                left = sum(1 for q in queue if q.project == task.project) + 1
                return (
                    0,
                    rank,
                    left,
                    task.project,
                    task.revision,
                    task.model,
                    task.run_number,
                )
            return _key

        def _key(task: Task, queue: deque[Task]) -> tuple:
            left = sum(1 for q in queue if q.project == task.project) + 1
            return (
                left,
                task.project,
                task.revision,
                task.model,
                task.run_number,
            )
        return _key

    if mode == "model":
        if prioritized:
            def _key(task: Task, queue: deque[Task]) -> tuple:
                del queue
                rank = prioritized_index.get(task.model)
                if rank is None:
                    return (
                        1,
                        task.model,
                        task.revision,
                        task.project,
                        task.run_number,
                    )
                return (
                    0,
                    rank,
                    task.model,
                    task.revision,
                    task.project,
                    task.run_number,
                )
            return _key

        def _key(task: Task, queue: deque[Task]) -> tuple:
            left = sum(1 for q in queue if q.model == task.model) + 1
            return (
                left,
                task.model,
                task.project,
                task.revision,
                task.run_number,
            )
        return _key

    if mode == "random":
        def _key(task: Task, _queue: deque[Task]) -> tuple:
            task_id = task_key(task)
            random_scores.setdefault(task_id, random.random())
            return (
                random_scores[task_id],
                task.project,
                task.revision,
                task.model,
                task.run_number,
            )
        return _key

    raise ValueError(f"Unsupported priority mode: {mode}")


def _merge_exclude_patterns(tasks_dir: pathlib.Path, revisions: list[str],
                            exclude: list[str]) -> list[str]:
    for rev in revisions:
        props = _load_props(tasks_dir, rev)
        exclude_raw = props.get("exclude", "")
        if exclude_raw:
            for p in (s.strip() for s in exclude_raw.split(",") if s.strip()):
                if p not in exclude:
                    exclude.append(p)
    return exclude


def _format_duration(ms: float) -> str:
    secs = int(ms / 1000)
    if secs < 60:
        return f"{secs}s"
    minutes = secs // 60
    remaining_secs = secs % 60
    if minutes < 60:
        return f"{minutes}m {remaining_secs}s"
    hours = minutes // 60
    remaining_mins = minutes % 60
    return f"{hours}h {remaining_mins}m"


def _get_cached_run_outcome(
    outcome_cache: dict[tuple[str, int, str, str], RunOutcome | None],
    results_root: pathlib.Path,
    project_name: str,
    run_number: int,
    model: str,
    revision: str,
) -> RunOutcome | None:
    key = (project_name, run_number, model, revision)
    if key in outcome_cache:
        return outcome_cache[key]
    path = _results_json_path(results_root, project_name, run_number, model, revision)
    outcome = _read_run_outcome_from_results_file(path)
    outcome_cache[key] = outcome
    return outcome


def _remaining_counts_by_model_for_run(
    outcome_cache: dict[tuple[str, int, str, str], RunOutcome | None],
    *,
    results_root: pathlib.Path,
    project_name: str,
    run_number: int,
    revisions: list[str],
    models: list[str],
    rerun_mode: str,
) -> tuple[dict[str, int] | None, bool]:
    """
    Returns (counts_by_model, exact).

    - counts_by_model=None means we cannot know anything for this run yet (display as '???').
    - exact=True means counts are exact.
    - exact=False means counts are lower bounds (display as '>= ...').

    Semantics:
      - run 1 is always exact: count missing results for run 1.
      - rerun=all: exact for any run N: count missing results for run N.
      - rerun=failed (N>1): count missing results for run N among items that are known
        to have failed run N-1. If run N-1 is not fully complete, this is a lower bound.
        If run N-1 has not started at all, counts are unknown (None).
    """
    if run_number == 1 or rerun_mode == "all":
        counts: dict[str, int] = {m: 0 for m in models}
        for rev in revisions:
            for model in models:
                outcome = _get_cached_run_outcome(
                    outcome_cache,
                    results_root,
                    project_name,
                    run_number,
                    model,
                    rev,
                )
                if outcome is None:
                    counts[model] += 1
        return counts, True

    prev_run = run_number - 1
    prev_any_present = False
    prev_complete = True
    counts = {m: 0 for m in models}

    for rev in revisions:
        for model in models:
            prev_outcome = _get_cached_run_outcome(
                outcome_cache,
                results_root,
                project_name,
                prev_run,
                model,
                rev,
            )
            if prev_outcome is None:
                prev_complete = False
                continue

            prev_any_present = True
            if prev_outcome != RunOutcome.SUCCESS:
                cur_outcome = _get_cached_run_outcome(
                    outcome_cache,
                    results_root,
                    project_name,
                    run_number,
                    model,
                    rev,
                )
                if cur_outcome is None:
                    counts[model] += 1

    if not prev_any_present:
        return None, False

    return counts, prev_complete


def _aggregate_remaining_counts_by_model_for_run(
    outcome_cache: dict[tuple[str, int, str, str], RunOutcome | None],
    *,
    project_entries: list[dict],
    results_root: pathlib.Path,
    run_number: int,
    models: list[str],
    rerun_mode: str,
) -> tuple[dict[str, int] | None, bool]:
    any_known = False
    exact_all = True
    total_counts: dict[str, int] = {m: 0 for m in models}

    for entry in project_entries:
        project_path: pathlib.Path = entry["project_path"]
        revisions: list[str] = entry["revisions"]
        counts, exact = _remaining_counts_by_model_for_run(
            outcome_cache,
            results_root=results_root,
            project_name=project_path.name,
            run_number=run_number,
            revisions=revisions,
            models=models,
            rerun_mode=rerun_mode,
        )
        if counts is None:
            exact_all = False
            continue

        any_known = True
        if not exact:
            exact_all = False
        for m in models:
            total_counts[m] += counts.get(m, 0)

    if not any_known:
        return None, False

    return total_counts, exact_all


def _historical_counts_by_model(
    *,
    project_entries: list[dict],
    models: list[str],
    runs: int,
    rerun_mode: str,
    results_root: pathlib.Path,
) -> tuple[dict[str, int], dict[str, int]]:
    completed_by_model: dict[str, int] = {m: 0 for m in models}
    success_by_model: dict[str, int] = {m: 0 for m in models}

    for entry in project_entries:
        project_path: pathlib.Path = entry["project_path"]
        revisions: list[str] = entry["revisions"]

        for revision in revisions:
            for model in models:
                previous_outcome: RunOutcome | None = None
                for run_number in range(1, runs + 1):
                    if not _should_run_in_run(rerun_mode, run_number, previous_outcome):
                        break

                    result_path = _results_json_path(results_root, project_path.name, run_number, model, revision)
                    if not result_path.exists():
                        break

                    outcome = _read_run_outcome_from_results_file(result_path)
                    completed_by_model[model] += 1
                    if outcome == RunOutcome.SUCCESS:
                        success_by_model[model] += 1
                    previous_outcome = outcome

    return completed_by_model, success_by_model


def _print_remaining_status_summary(
    *,
    project_entries: list[dict],
    models: list[str],
    runs: int,
    rerun_mode: str,
    results_root: pathlib.Path,
    project_label: str | None,
    avg_elapsed_ms: float = 0.0,
    threads: int = 1,
    output: _BprOutput | None = None,
) -> None:
    output = _resolve_output(output)
    outcome_cache: dict[tuple[str, int, str, str], RunOutcome | None] = {}

    for run_number in range(1, runs + 1):
        counts_by_model, exact = _aggregate_remaining_counts_by_model_for_run(
            outcome_cache,
            project_entries=project_entries,
            results_root=results_root,
            run_number=run_number,
            models=models,
            rerun_mode=rerun_mode,
        )
        unknown = counts_by_model is None

        _report_tasks_remaining_for_run(
            tasks=[],
            models=models,
            run_number=run_number,
            project_label=project_label if run_number == 1 else None,
            avg_elapsed_ms=avg_elapsed_ms,
            threads=threads,
            exact=exact,
            unknown=unknown,
            counts_by_model_override=counts_by_model,
            output=output,
        )


def _run_single_project(
    args: argparse.Namespace,
    models: list[str],
    base_cli_tokens: list[str],
    results_root: pathlib.Path,
    jvm_args: list[str],
    filter_revisions: set[str] | None,
    *,
    output: _BprOutput | None = None,
    task_scheduled: TaskScheduledCallback | None = None,
    task_started: TaskStartedCallback | None = None,
    task_completed: TaskCompletedCallback | None = None,
    history_seeded: TaskHistorySeededCallback | None = None,
    task_worktrees: TaskWorktreeMap | None = None,
    exit_fn: ExitFn = sys.exit,
) -> int:
    output = _resolve_output(output)
    project_path = pathlib.Path(args.project).resolve()
    tasks_dir = pathlib.Path(args.tasksdir).resolve()

    job_env: dict[str, str] | None = None
    heap_mb = 1024
    dataset_threads = 20
    try:
        config = get_dataset_config(project_path.name, args.commits_dir)
        job_env = build_job_env(config)
        job_env["BRK_MODE"] = args.mode
        heap_mb = config.heap_mb
        dataset_threads = config.threads
    except ValueError:
        pass

    revisions = _read_revisions_from_lines(sys.stdin)
    revisions = _apply_revision_filter(revisions, filter_revisions)

    existing_revisions = [rev for rev in revisions if (tasks_dir / f"{rev}.txt").exists()]
    if len(revisions) != len(existing_revisions):
        output.stderr(
            f"Warning: {len(revisions)} revisions requested, but only {len(existing_revisions)} tasks found on disk",
        )
    revisions = existing_revisions

    if not revisions:
        output.stderr("No valid revisions with task files provided on stdin.")
        _write_exit(3, exit_fn)

    valid_revisions = _validate_revisions(project_path, revisions, output=output)
    if len(valid_revisions) != len(revisions):
        dropped = set(revisions) - set(valid_revisions)
        output.stderr("Validation failed for revisions: " + ", ".join(dropped))
        _write_exit(4, exit_fn)
    revisions = valid_revisions

    avg_llm_millis = _compute_avg_llm_millis(results_root, project_path.name, revisions, args.runs)
    task_priority_key = _task_priority_key(
        args.priority,
        llm_scores={(str(project_path), rev): avg_llm_millis.get(rev, 0.0) for rev in revisions},
    )

    exclude = _merge_exclude_patterns(tasks_dir, revisions, args.exclude or [])

    threads = args.threads or dataset_threads
    avg_elapsed = _compute_avg_elapsed_millis(results_root, project_path.name, args.runs)

    commit_tests_cb = make_commit_tests(exclude)
    execute_tests_cb = make_execute_tests()

    def get_cli_args_by_task(task: Task) -> list[str]:
        args_list, _edits, _tests = _build_cli_args_for_job(
            project_path=project_path,
            tasks_dir=tasks_dir,
            revision=task.revision,
            model=task.model,
            planmodel=args.planmodel,
            base_cli_tokens=base_cli_tokens,
            exclude_patterns=exclude,
            mode=args.mode,
        )
        return args_list

    project_entries = [
        {
            "project_path": project_path,
            "revisions": revisions,
        }
    ]
    _print_remaining_status_summary(
        project_entries=project_entries,
        models=models,
        runs=args.runs,
        rerun_mode=args.rerun,
        results_root=results_root,
        project_label=project_path.name,
        avg_elapsed_ms=avg_elapsed,
        threads=threads,
        output=output,
    )
    if history_seeded is not None:
        completed_by_model, success_by_model = _historical_counts_by_model(
            project_entries=project_entries,
            models=models,
            runs=args.runs,
            rerun_mode=args.rerun,
            results_root=results_root,
        )
        history_seeded(completed_by_model, success_by_model)

    if args.dry_run:
        _write_exit(0, exit_fn)

    props_by_rev: dict[str, dict[str, str]] = {rev: _load_props(tasks_dir, rev) for rev in revisions}

    initial_tasks: list[Task] = []
    for rev in revisions:
        for model in models:
            prev_outcome: RunOutcome | None = None
            for run_number in range(1, args.runs + 1):
                if not _should_run_in_run(args.rerun, run_number, prev_outcome):
                    break

                cur_path = _results_json_path(results_root, project_path.name, run_number, model, rev)
                if cur_path.exists():
                    prev_outcome = _read_run_outcome_from_results_file(cur_path)
                    continue

                task_obj = Task(
                    project=str(project_path),
                    revision=rev,
                    model=model,
                    run_number=run_number,
                    job_env=job_env,
                    heap_mb=heap_mb,
                    properties=props_by_rev[rev],
                )
                if task_worktrees is not None:
                    task_worktrees.mark_scheduled(task_obj)
                if task_scheduled is not None:
                    task_scheduled(task_obj)
                initial_tasks.append(task_obj)
                break

    def _task_started(task: Task, worktree_path: pathlib.Path, attempt: int) -> None:
        if task_worktrees is not None:
            task_worktrees.mark_started(task, worktree_path)
        if attempt != 1:
            return
        if task_started is not None:
            task_started(task, worktree_path, attempt)

    def _on_task_complete(task: Task, result: RunResult) -> list[Task]:
        try:
            if task_completed is not None:
                task_completed(task, result)
        finally:
            if task_worktrees is not None:
                task_worktrees.mark_completed(task)

        if result.outcome == RunOutcome.AGENT_ERROR:
            return []
        if task.run_number >= args.runs:
            return []

        if args.rerun != "all" and result.outcome == RunOutcome.SUCCESS:
            return []

        next_run = task.run_number + 1
        next_path = _results_json_path(results_root, project_path.name, next_run, task.model, task.revision)
        if next_path.exists():
            return []

        next_task = Task(
            project=task.project,
            revision=task.revision,
            model=task.model,
            run_number=next_run,
            job_env=task.job_env,
            heap_mb=task.heap_mb,
            properties=task.properties,
        )
        if task_worktrees is not None:
            task_worktrees.mark_scheduled(next_task)
        if task_scheduled is not None:
            task_scheduled(next_task)
        return [next_task]

    results_map_all = run_pipelined_tasks(
        initial_tasks=initial_tasks,
        results_root=results_root,
        threads=threads,
        jvm_args=jvm_args,
        stagger_seconds=args.stagger_seconds,
        get_cli_args=get_cli_args_by_task,
        execute_tests=execute_tests_cb,
        commit_tests=commit_tests_cb,
        on_task_complete=_on_task_complete,
        on_task_start=_task_started,
        max_heap_mb=args.maxheap,
        task_priority=task_priority_key,
    )

    return _summarize_and_exit(results_map_all, output=output, exit_fn=exit_fn)


def _run_multi_project(
    args: argparse.Namespace,
    models: list[str],
    base_cli_tokens: list[str],
    results_root: pathlib.Path,
    jvm_args: list[str],
    filter_revisions: set[str] | None,
    *,
    output: _BprOutput | None = None,
    task_scheduled: TaskScheduledCallback | None = None,
    task_started: TaskStartedCallback | None = None,
    task_completed: TaskCompletedCallback | None = None,
    history_seeded: TaskHistorySeededCallback | None = None,
    task_worktrees: TaskWorktreeMap | None = None,
    exit_fn: ExitFn = sys.exit,
) -> int:
    output = _resolve_output(output)
    from dataset_config import DEFAULT_DATASETS

    if args.projects:
        dataset_names = [d.strip() for d in args.projects.split(",") if d.strip()]
    else:
        dataset_names = DEFAULT_DATASETS

    tasks_dir = pathlib.Path(args.tasksdir).resolve()
    commits_dir = args.commits_dir

    all_exclude_patterns: list[str] = list(args.exclude or [])
    avg_llm_scores: dict[tuple[str, str], float] = {}

    project_entries: list[dict] = []
    total_thread_sum = 0

    cli_args_dispatchers: dict[str, Callable[[Task], list[str]]] = {}

    for ds_name in dataset_names:
        try:
            config = get_dataset_config(ds_name, commits_dir)
        except ValueError as e:
            output.stderr(f"Error: {e}")
            _write_exit(1, exit_fn)

        project_path = pathlib.Path(config.project_path).resolve()
        if not project_path.is_dir():
            output.stderr(f"Error: project directory {project_path} does not exist for dataset '{ds_name}'")
            _write_exit(1, exit_fn)

        if not pathlib.Path(config.commits_file).is_file():
            output.stderr(f"Error: commits file {config.commits_file} not found for dataset '{ds_name}'")
            _write_exit(1, exit_fn)

        revisions = _read_revisions_from_file(config.commits_file)
        revisions = _apply_revision_filter(revisions, filter_revisions)
        existing_revisions = [rev for rev in revisions if (tasks_dir / f"{rev}.txt").exists()]
        if len(revisions) != len(existing_revisions):
            output.stderr(
                f"Warning ({ds_name}): {len(revisions)} revisions requested, "
                f"but only {len(existing_revisions)} tasks found on disk",
            )
        revisions = existing_revisions

        if not revisions:
            output.stderr(f"Warning ({ds_name}): no valid revisions, skipping")
            continue

        revisions = _validate_revisions(project_path, revisions, output=output)
        if not revisions:
            output.stderr(f"Warning ({ds_name}): all revisions failed validation, skipping")
            continue

        avg_llm_millis = _compute_avg_llm_millis(results_root, project_path.name, revisions, args.runs)
        for revision in revisions:
            avg_llm_scores[(str(project_path), revision)] = avg_llm_millis.get(revision, 0.0)
        revisions.sort(key=lambda r: avg_llm_millis.get(r, 0.0), reverse=True)

        exclude = _merge_exclude_patterns(tasks_dir, revisions, list(args.exclude or []))
        for p in exclude:
            if p not in all_exclude_patterns:
                all_exclude_patterns.append(p)

        job_env = build_job_env(config)
        job_env["BRK_MODE"] = args.mode
        project_str = str(project_path)

        def _make_cli_args_fn(
            pp: pathlib.Path,
            td: pathlib.Path,
            pm: str | None,
            bt: list[str],
            ex: list[str],
            mode: str,
        ) -> Callable[[Task], list[str]]:
            def _fn(task: Task) -> list[str]:
                args_list, _, _ = _build_cli_args_for_job(
                    project_path=pp,
                    tasks_dir=td,
                    revision=task.revision,
                    model=task.model,
                    planmodel=pm,
                    base_cli_tokens=bt,
                    exclude_patterns=ex,
                    mode=mode,
                )
                return args_list

            return _fn

        cli_args_dispatchers[project_str] = _make_cli_args_fn(
            project_path, tasks_dir, args.planmodel, base_cli_tokens, exclude, args.mode
        )

        project_entries.append(
            {
                "dataset": ds_name,
                "project_path": project_path,
                "revisions": revisions,
                "config": config,
                "job_env": job_env,
                "exclude": exclude,
                "avg_llm_millis": avg_llm_millis,
            }
        )

        total_thread_sum += config.threads

    threads = args.threads if args.threads is not None else (total_thread_sum or 20)

    _print_remaining_status_summary(
        project_entries=project_entries,
        models=models,
        runs=args.runs,
        rerun_mode=args.rerun,
        results_root=results_root,
        project_label="(all projects)",
        threads=threads,
        output=output,
    )
    if history_seeded is not None:
        completed_by_model, success_by_model = _historical_counts_by_model(
            project_entries=project_entries,
            models=models,
            runs=args.runs,
            rerun_mode=args.rerun,
            results_root=results_root,
        )
        history_seeded(completed_by_model, success_by_model)

    if args.dry_run:
        _write_exit(0, exit_fn)

    if not project_entries:
        output.stderr("All provided revision/model combinations have already been processed.")
        _write_exit(0, exit_fn)

    commit_tests_cb = make_commit_tests(all_exclude_patterns)
    execute_tests_cb = make_execute_tests()

    def dispatch_cli_args(task: Task) -> list[str]:
        project_str = str(pathlib.Path(task.project).resolve())
        fn = cli_args_dispatchers.get(project_str)
        if fn is None:
            raise ValueError(f"No CLI args dispatcher for project {task.project}")
        return fn(task)

    # Printed for both dry-run and execute paths above; keep execute startup output consistent.

    props_cache: dict[str, dict[str, str]] = {}

    def _props_for_rev(rev: str) -> dict[str, str]:
        existing = props_cache.get(rev)
        if existing is not None:
            return existing
        props_cache[rev] = _load_props(tasks_dir, rev)
        return props_cache[rev]

    initial_tasks_with_sort: list[tuple[float, Task]] = []
    for entry in project_entries:
        project_path: pathlib.Path = entry["project_path"]
        revisions: list[str] = entry["revisions"]
        config = entry["config"]
        avg_llm_millis: dict[str, float] = entry["avg_llm_millis"]

        for rev in revisions:
            for model in models:
                prev_outcome: RunOutcome | None = None
                for run_number in range(1, args.runs + 1):
                    if not _should_run_in_run(args.rerun, run_number, prev_outcome):
                        break

                    cur_path = _results_json_path(results_root, project_path.name, run_number, model, rev)
                    if cur_path.exists():
                        prev_outcome = _read_run_outcome_from_results_file(cur_path)
                        continue

                    task = Task(
                        project=str(project_path),
                        revision=rev,
                        model=model,
                        run_number=run_number,
                        job_env=entry["job_env"],
                        heap_mb=config.heap_mb,
                        properties=_props_for_rev(rev),
                    )
                    if task_worktrees is not None:
                        task_worktrees.mark_scheduled(task)
                    if task_scheduled is not None:
                        task_scheduled(task)
                    initial_tasks_with_sort.append((avg_llm_millis.get(rev, 0.0), task))
                    break

    initial_tasks = [t for _, t in sorted(initial_tasks_with_sort, key=lambda p: p[0], reverse=True)]
    task_priority_key = _task_priority_key(args.priority, llm_scores=avg_llm_scores)

    def _task_started(task: Task, worktree_path: pathlib.Path, attempt: int) -> None:
        if task_worktrees is not None:
            task_worktrees.mark_started(task, worktree_path)
        if attempt != 1:
            return
        if task_started is not None:
            task_started(task, worktree_path, attempt)

    def _on_task_complete(task: Task, result: RunResult) -> list[Task]:
        try:
            if task_completed is not None:
                task_completed(task, result)
        finally:
            if task_worktrees is not None:
                task_worktrees.mark_completed(task)

        if result.outcome == RunOutcome.AGENT_ERROR:
            return []
        if task.run_number >= args.runs:
            return []

        if args.rerun != "all" and result.outcome == RunOutcome.SUCCESS:
            return []

        project_path = pathlib.Path(task.project)
        next_run = task.run_number + 1
        next_path = _results_json_path(results_root, project_path.name, next_run, task.model, task.revision)
        if next_path.exists():
            return []

        next_task = Task(
            project=task.project,
            revision=task.revision,
            model=task.model,
            run_number=next_run,
            job_env=task.job_env,
            heap_mb=task.heap_mb,
            properties=task.properties,
        )
        if task_worktrees is not None:
            task_worktrees.mark_scheduled(next_task)
        if task_scheduled is not None:
            task_scheduled(next_task)
        return [next_task]

    results_map_all = run_pipelined_tasks(
        initial_tasks=initial_tasks,
        results_root=results_root,
        threads=threads,
        jvm_args=jvm_args,
        stagger_seconds=args.stagger_seconds,
        get_cli_args=dispatch_cli_args,
        execute_tests=execute_tests_cb,
        commit_tests=commit_tests_cb,
        on_task_complete=_on_task_complete,
        on_task_start=_task_started,
        max_heap_mb=args.maxheap,
        task_priority=task_priority_key,
    )

    return _summarize_and_exit(results_map_all, output=output, exit_fn=exit_fn)



def _summarize_and_exit(
    results_map: dict[tuple[str, str, str, int], RunResult],
    *,
    output: _BprOutput | None = None,
    exit_fn: ExitFn = sys.exit,
) -> int:
    output = _resolve_output(output)
    outcomes = [res.outcome for res in results_map.values()]

    agent_error_keys = [k for k, v in results_map.items() if v.outcome == RunOutcome.AGENT_ERROR]
    if agent_error_keys:
        output.stderr("Agent encountered internal errors for some tasks.")
        output.stderr("Agent-errored tasks (review):")
        for project, rev, model, run_number in sorted(agent_error_keys):
            output.stderr(f"  run={run_number} model={model} rev={rev} project={project}")
        _write_exit(2, exit_fn)

    if any(o == RunOutcome.AGENT_FAILED for o in outcomes):
        output.stderr("Agent failed to find a solution for some revisions.")
        _write_exit(0, exit_fn)
    if any(o == RunOutcome.TESTS_FAILED for o in outcomes):
        output.stderr("Tests failed for some revisions.")
        _write_exit(0, exit_fn)

    output.stdout("All revisions processed successfully!")
    _write_exit(0, exit_fn)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brokk BPR runner (front-end).")
    project_group = parser.add_mutually_exclusive_group(required=False)
    project_group.add_argument("--project", help="Git project directory (single-project mode, reads revisions from stdin).")
    project_group.add_argument(
        "--projects",
        help="Comma-separated dataset names for multi-project mode (e.g. brokk,kafka,ruff). Defaults to all datasets.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Comma-separated list of model names to pass to the agent (one job per model).",
    )
    parser.add_argument(
        "--planmodel",
        help="Override the planning model to use. If omitted, uses the value from --model.",
    )
    parser.add_argument(
        "--priority",
        default=_parse_priority_spec("hardest"),
        type=_parse_priority_spec,
        help=(
            "Task scheduling priority: hardest (default), easiest, dataset, model, random, "
            "or dataset=<dataset1,dataset2>, model=<model1,model2>."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["code", "agent", "infer-context"],
        help="Benchmark mode (results written to <mode>results/).",
    )
    parser.add_argument(
        "--tasksdir",
        default="codetasks",
        help="Directory that contains task files (<rev>.txt) and optional <rev>.properties.",
    )
    parser.add_argument(
        "--filter",
        dest="filter_file",
        help="Path to a JSONL file with rows like {'hash': '...'} or {'hash': '...', 'difficulty': 0.3}; only listed hashes are run.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of parallel threads (default: 20 for single-project, sum of dataset defaults for multi-project).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run each revision/model combination.",
    )
    parser.add_argument(
        "--rerun",
        choices=["failed", "all"],
        default="failed",
        help="For run N>1: 'failed' reruns only tasks that failed in run N-1; 'all' reruns all tasks. Default: failed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show task counts and exit without executing runs.",
    )
    parser.add_argument(
        "--zero-remaining",
        action="store_true",
        help="Fill missing task/model run result files with MERCY_RULE placeholders and exit.",
    )
    parser.add_argument(
        "--stagger",
        dest="stagger_seconds",
        type=int,
        default=2,
        help="Add a random 0-Ns sleep before starting each task (default: 2).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Filenames or glob patterns to exclude from --edit options. May be supplied multiple times.",
    )
    parser.add_argument(
        "--cli-dir",
        default="../brokk/",
        help="Path to the Brokk CLI directory (default: %(default)s, resolves to <dir>/cli).",
    )
    parser.add_argument(
        "--results-dir",
        help="Override results root directory (default: <mode>results<cli_suffix>/).",
    )
    parser.add_argument(
        "--cli-args",
        default="",
        help="Additional CLI arguments (string) to pass to the agent (parsed with shlex). Do not include --project/--worktree.",
    )
    parser.add_argument(
        "--commits-dir",
        default="taskcommits-0126",
        help="Directory containing per-project commit files (multi-project mode only).",
    )
    parser.add_argument(
        "--maxheap",
        type=int,
        default=None,
        help="Maximum total heap (MB) across concurrent tasks. Limits concurrency based on per-task -Xmx.",
    )
    return parser


def run_with_args(
    args: argparse.Namespace,
    unknown_args: list[str],
    *,
    output: _BprOutput | None = None,
    task_scheduled: TaskScheduledCallback | None = None,
    task_started: TaskStartedCallback | None = None,
    task_completed: TaskCompletedCallback | None = None,
    history_seeded: TaskHistorySeededCallback | None = None,
    task_worktrees: TaskWorktreeMap | None = None,
    exit_fn: ExitFn = sys.exit,
) -> int:
    output = _resolve_output(output)
    # Forward JVM-style flags to the CLI; error on anything else.
    jvm_args: list[str] = [ua for ua in unknown_args if ua.startswith("-X") or ua.startswith("-D")]
    invalid_unknowns = [ua for ua in unknown_args if ua not in jvm_args]
    if invalid_unknowns:
        output.stderr(f"Unknown arguments: {' '.join(invalid_unknowns)}")
        _write_exit(5, exit_fn)
        return 5

    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if not models:
        output.stderr("Error: --model must contain at least one model name")
        _write_exit(1, exit_fn)
        return 1

    filter_path = pathlib.Path(args.filter_file).resolve() if args.filter_file else None
    filter_revisions = _load_filter_revisions(filter_path, output=output, exit_fn=exit_fn)

    results_root = pathlib.Path(args.results_dir) if args.results_dir else pathlib.Path(f"{args.mode}results")
    if args.zero_remaining:
        return _run_zero_remaining(
            args,
            models,
            results_root,
            filter_revisions,
            output=output,
            exit_fn=exit_fn,
        )

    # Validate API key is present
    validate_api_key()
    set_cli_bin(pathlib.Path(args.cli_dir))
    validate_models(models)
    base_cli_tokens = shlex.split(args.cli_args or "")

    if args.project:
        return _run_single_project(
            args,
            models,
            base_cli_tokens,
            results_root,
            jvm_args,
            filter_revisions,
            output=output,
            task_scheduled=task_scheduled,
            task_started=task_started,
            task_completed=task_completed,
            history_seeded=history_seeded,
            task_worktrees=task_worktrees,
            exit_fn=exit_fn,
        )

    return _run_multi_project(
        args,
        models,
        base_cli_tokens,
        results_root,
        jvm_args,
        filter_revisions,
        output=output,
        task_scheduled=task_scheduled,
        task_started=task_started,
        task_completed=task_completed,
        history_seeded=history_seeded,
        task_worktrees=task_worktrees,
        exit_fn=exit_fn,
    )


def main() -> None:
    # Load environment variables from .env file if present
    load_dotenv()
    validate_cache_dirs()

    parser = _build_parser()
    args, unknown_args = parser.parse_known_args()
    run_with_args(args, unknown_args)

if __name__ == "__main__":
    main()
