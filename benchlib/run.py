import concurrent.futures
import dataclasses
import enum
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from collections.abc import Callable

import benchlib.cli
from benchlib.cli import get_cli_info, run_cli
from benchlib.git import git_run
from benchlib.worktree import archive_worktree


def atomic_write_json(path: pathlib.Path, data: dict) -> None:
    """Write JSON atomically using temp file + rename."""
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".json.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.rename(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


class RunOutcome(enum.Enum):
    SUCCESS = 0
    AGENT_ERROR = 1
    AGENT_FAILED = 2
    TESTS_FAILED = 3


@dataclasses.dataclass
class Task:
    project: str
    revision: str
    model: str
    run_number: int
    job_env: dict[str, str] | None = None
    heap_mb: int = 1024
    properties: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RunResult:
    outcome: RunOutcome


class HeapBudget:
    def __init__(self, max_mb: int):
        self._max = max_mb
        self._used = 0
        self._cond = threading.Condition()

    def acquire(self, mb: int) -> None:
        with self._cond:
            while self._used + mb > self._max:
                self._cond.wait()
            self._used += mb

    def release(self, mb: int) -> None:
        with self._cond:
            self._used -= mb
            self._cond.notify_all()


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


def _create_worktree_session_name(model: str, revshort: str, run_number: int) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"{model}-{revshort}-{run_number}-{ts}"


def _archive_worktree(project_path: pathlib.Path, worktree_path: pathlib.Path) -> None:
    archive_worktree(project_path, worktree_path)


def _run_one_job(
    task: Task,
    project_path: pathlib.Path,
    results_root: pathlib.Path,
    jvm_args: list[str],
    stagger_seconds: int,
    get_cli_args: Callable[[Task], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path, dict[str, str], dict[str, str]], subprocess.CompletedProcess],
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str, dict[str, str]], None],
    on_task_start: Callable[[Task, pathlib.Path, int], None] | None = None,
    attempt: int = 1,
) -> RunResult:
    run_env = task.job_env.copy() if task.job_env else os.environ.copy()
    run_env["BRK_COLLECT_METRICS"] = "true"

    # Explicitly enable context cache for infer-context mode
    if run_env.get("BRK_MODE") == "infer-context":
        run_env["BRK_CONTEXT_CACHE"] = "RW"

    if stagger_seconds > 0:
        time.sleep(random.uniform(0, stagger_seconds))

    revshort = git_run(project_path, "rev-parse", "--short", task.revision).strip()

    run_id = f"{task.run_number}/{revshort}"
    start_time = datetime.now()
    start_monotonic = time.monotonic()

    workdir_name = _create_worktree_session_name(task.model, revshort, task.run_number)

    worktree_root = run_env.get("BRK_WORKTREE_ROOT")
    if worktree_root:
        worktree_base = pathlib.Path(worktree_root)
        if run_env.get("BB_DEBUG"):
            print(f"Using custom worktree root: {worktree_base}", file=sys.stderr)
    else:
        worktree_base = pathlib.Path.home() / "brokkbench"

    worktree_path = worktree_base / project_path.name / workdir_name
    if on_task_start is not None:
        try:
            on_task_start(task, worktree_path, attempt)
        except Exception:
            pass
    print(f"BEGIN {run_id} {start_time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)

    stop_reason: str = "UNKNOWN"
    metrics: dict | None = None
    metrics_elapsed_ms: int | None = None
    context_added: int | None = None
    context_removed: int | None = None

    def _wall_elapsed_ms() -> int:
        return int((time.monotonic() - start_monotonic) * 1000)

    try:
        os.makedirs(worktree_path.parent, exist_ok=True)
        agent_log_path = worktree_path.parent / f"{workdir_name}-agent.txt"
        run_output_path = worktree_path / "run-output.txt"
        prebuild_log_path = worktree_path / "prebuild.txt"
        harness_log_path = worktree_path.parent / f"{workdir_name}-harness-tests.txt"
        def _write_run_output(paths: list[pathlib.Path]) -> None:
            worktree_path.mkdir(parents=True, exist_ok=True)
            with open(run_output_path, "wb") as out_fp:
                for p in paths:
                    if not p.exists():
                        continue
                    out_fp.write(f"-------------- {p}\n".encode())
                    out_fp.flush()
                    with open(p, "rb") as src_fp:
                        shutil.copyfileobj(src_fp, out_fp)
                    out_fp.write(b"\n")

        def _results_path() -> pathlib.Path:
            results_dir = results_root / f"{project_path.name}{task.run_number}"
            results_dir.mkdir(parents=True, exist_ok=True)
            return results_dir / f"{task.model}-{task.revision}.json"

        def persist_metrics() -> None:
            if metrics is not None:
                atomic_write_json(_results_path(), metrics)
            _write_run_output([agent_log_path, prebuild_log_path, harness_log_path])

        # Step 1: Create worktree
        first_cmd = [
            str(benchlib.cli.CLI_BIN),
            *jvm_args,
            f"--project={project_path}",
            f"--worktree={worktree_path}",
        ]
        if run_cli(first_cmd, agent_log_path, run_env).returncode != 0:
            stop_reason = "AGENT_ERROR"
            persist_metrics()
            return RunResult(outcome=RunOutcome.AGENT_ERROR)

        # Step 2: Reset to revision^
        git_run(worktree_path, "reset", "--hard", f"{task.revision}^")

        # Step 3: Commit tests (caller-provided callback)
        commit_tests(project_path, worktree_path, task.revision, run_env)

        # Step 4: Build and run CLI command
        cli_args = get_cli_args(task)
        second_cmd = [
            str(benchlib.cli.CLI_BIN),
            *jvm_args,
            f"--project={project_path}",
            f"--worktree={worktree_path}",
            *cli_args,
        ]
        if run_cli(second_cmd, agent_log_path, run_env).returncode != 0:
            stop_reason = "AGENT_ERROR"
            persist_metrics()
            return RunResult(outcome=RunOutcome.AGENT_ERROR)

        # Step 5: Extract metrics (use the last matching line; multiple can occur)
        def _extract_last_metrics_payload(log_path: pathlib.Path, prefix: str) -> str | None:
            payload: str | None = None
            with open(log_path, "r", encoding="utf-8") as log_fp:
                for line in log_fp:
                    stripped = line.strip()
                    if stripped.startswith(prefix):
                        payload = stripped[len(prefix):]
            return payload

        codeagent_payload = _extract_last_metrics_payload(agent_log_path, "BRK_CODEAGENT_METRICS=")
        context_payload = _extract_last_metrics_payload(agent_log_path, "BRK_CONTEXT_METRICS=")

        if codeagent_payload is None:
            stop_reason = "AGENT_ERROR"
            persist_metrics()
            return RunResult(outcome=RunOutcome.AGENT_ERROR)

        metrics = json.loads(codeagent_payload)
        metrics["worktree"] = str(worktree_path)

        if context_payload is not None:
            try:
                context_metrics = json.loads(context_payload)
                metrics["contextMetrics"] = context_metrics
                added = context_metrics.get("addedFragments")
                removed = context_metrics.get("removedFragments")
                if isinstance(added, int) and isinstance(removed, int):
                    metrics["contextDiffCount"] = added + removed
                    context_added = added
                    context_removed = removed
            except Exception:
                pass

        elapsed_val = metrics.get("elapsedMillis")
        if isinstance(elapsed_val, (int, float)):
            metrics_elapsed_ms = int(elapsed_val)

        cli_info = get_cli_info()
        metrics["cliVersion"] = cli_info["cliVersion"]
        metrics["proxy"] = cli_info["proxy"]

        stop_reason_val = metrics.get("stopReason")
        if isinstance(stop_reason_val, str) and stop_reason_val:
            stop_reason = stop_reason_val

        # Step 6: Commit agent work
        git_run(worktree_path, "add", "-A")
        git_run(worktree_path, "commit", "--allow-empty", "-m", "Agent work")

        if stop_reason != "SUCCESS":
            persist_metrics()
            return RunResult(outcome=RunOutcome.AGENT_FAILED)

        # Step 6b: Check that the agent actually made edits; if not, mark NO_EDITS and skip tests.
        diff_stat = git_run(worktree_path, "diff", "--name-only", "HEAD^", "HEAD")
        if not diff_stat.strip():
            stop_reason = "NO_EDITS"
            metrics["stopReason"] = stop_reason
            persist_metrics()
            return RunResult(outcome=RunOutcome.AGENT_FAILED)

        # Steps 7b and 8 only run when the agent reported SUCCESS,
        # to verify it did not cheat.

        # Step 7b: Optional pre-build
        prebuild_cmd = run_env.get("BRK_PREBUILD_CMD")
        if prebuild_cmd:
            if run_env.get("BB_DEBUG"):
                print(f"Running pre-build: {prebuild_cmd}", file=sys.stderr)
            with open(prebuild_log_path, "wb") as prebuild_log:
                prebuild_log.write(f"Running pre-build: {prebuild_cmd}\n".encode())
                prebuild_result = subprocess.run(
                    prebuild_cmd,
                    shell=True,
                    stdout=prebuild_log,
                    stderr=subprocess.STDOUT,
                    cwd=worktree_path,
                    env=run_env,
                )
                if prebuild_result.returncode != 0:
                    prebuild_log.write(f"\n-------------- exit code {prebuild_result.returncode}\n".encode())
                    prebuild_log.flush()
                    stop_reason = "PREBUILD_FAILED"
                    metrics["stopReason"] = stop_reason
                    persist_metrics()
                    return RunResult(outcome=RunOutcome.TESTS_FAILED)

        # Step 8: Execute tests (caller-provided callback)
        test_proc = execute_tests(project_path, worktree_path, run_env, task.properties)
        tests_failed = test_proc.returncode != 0

        with open(harness_log_path, "ab") as test_log:
            test_log.write(f"\n-------------- exit code {test_proc.returncode}\n".encode())

        if tests_failed:
            stop_reason = "HARNESS_TESTS_FAILED"
            metrics["stopReason"] = stop_reason

        # Step 9: Finalize metrics with total wall time
        total_elapsed_ms = _wall_elapsed_ms()
        metrics["elapsedMillis"] = total_elapsed_ms
        metrics_elapsed_ms = total_elapsed_ms

        persist_metrics()

        if tests_failed:
            return RunResult(outcome=RunOutcome.TESTS_FAILED)
        return RunResult(outcome=RunOutcome.SUCCESS)
    finally:
        elapsed_ms = metrics_elapsed_ms if metrics_elapsed_ms is not None else _wall_elapsed_ms()
        duration_str = _format_duration(elapsed_ms)
        context_suffix = ""
        if context_added is not None and context_removed is not None:
            context_suffix = f" +{context_added} -{context_removed}"
        print(f"COMPLETE {run_id} {stop_reason} {duration_str}{context_suffix} {worktree_path}", file=sys.stderr)
        try:
            if worktree_path.exists():
                _archive_worktree(project_path, worktree_path)
        except Exception as exc:
            print(f"Warning: failed to archive worktree '{worktree_path}': {exc}", file=sys.stderr)


def _run_with_retries(
    task: Task,
    project_path: pathlib.Path,
    results_root: pathlib.Path,
    jvm_args: list[str],
    stagger_seconds: int,
    get_cli_args: Callable[[Task], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path, dict[str, str], dict[str, str]], subprocess.CompletedProcess],
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str, dict[str, str]], None],
    on_task_start: Callable[[Task, pathlib.Path, int], None] | None = None,
) -> RunResult:
    MAX_ATTEMPTS = 3

    last_result = RunResult(outcome=RunOutcome.AGENT_ERROR)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = _run_one_job(
            task, project_path, results_root, jvm_args, stagger_seconds,
            get_cli_args, execute_tests, commit_tests,
            on_task_start=on_task_start,
            attempt=attempt,
        )
        last_result = result

        results_dir = results_root / f"{project_path.name}{task.run_number}"
        results_path = results_dir / f"{task.model}-{task.revision}.json"

        metrics = None
        if results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as fp:
                    metrics = json.load(fp)
            except Exception:
                metrics = None

        should_retry = False
        reason = None

        if metrics is None:
            should_retry = True
            reason = "no metrics file"
        else:
            stop_expl = str(metrics.get("stopExplanation", "")).lower()
            stop_reason = str(metrics.get("stopReason", ""))
            if (
                stop_reason == "LLM_ERROR"
                or "too many open files" in stop_expl
                or "check litellm logs" in stop_expl
                or "ratelimiterror" in stop_expl
            ):
                should_retry = True
                reason = stop_expl or stop_reason

        if should_retry and attempt < MAX_ATTEMPTS:
            print(f"Automatically retrying (attempt {attempt + 1}/{MAX_ATTEMPTS}) based on {reason}")
            continue

        break

    return last_result


def run_many_tasks(
    tasks: list[Task],
    results_root: pathlib.Path,
    threads: int,
    jvm_args: list[str],
    stagger_seconds: int,
    get_cli_args: Callable[[Task], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path, dict[str, str], dict[str, str]], subprocess.CompletedProcess],
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str, dict[str, str]], None],
    on_task_start: Callable[[Task, pathlib.Path, int], None] | None = None,
    max_heap_mb: int | None = None,
) -> dict[tuple[str, str, str, int], RunResult]:
    results_map: dict[tuple[str, str, str, int], RunResult] = {}
    heap_budget = HeapBudget(max_heap_mb) if max_heap_mb is not None else None

    def _heap_aware_wrapper(task, project_path, *args):
        weight = task.heap_mb
        heap_budget.acquire(weight)
        try:
            return _run_with_retries(
                task,
                project_path,
                *args,
                on_task_start=on_task_start,
            )
        finally:
            heap_budget.release(weight)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_job: dict[concurrent.futures.Future, tuple[str, str, str, int]] = {}
        for task in tasks:
            project_path = pathlib.Path(task.project).resolve()
            if heap_budget is not None:
                future = executor.submit(
                    _heap_aware_wrapper,
                    task, project_path, results_root, jvm_args, stagger_seconds,
                    get_cli_args, execute_tests, commit_tests,
                )
            else:
                future = executor.submit(
                    _run_with_retries,
                    task, project_path, results_root, jvm_args, stagger_seconds,
                    get_cli_args, execute_tests, commit_tests,
                    on_task_start=on_task_start,
                )
            future_to_job[future] = (task.project, task.revision, task.model, task.run_number)

        for future in concurrent.futures.as_completed(future_to_job):
            key = future_to_job[future]
            result = future.result()
            results_map[key] = result

    return results_map


def run_pipelined_tasks(
    initial_tasks: list[Task],
    results_root: pathlib.Path,
    threads: int,
    jvm_args: list[str],
    stagger_seconds: int,
    get_cli_args: Callable[["Task"], list[str]],
    execute_tests: Callable[[pathlib.Path, pathlib.Path, dict[str, str], dict[str, str]], subprocess.CompletedProcess],
    commit_tests: Callable[[pathlib.Path, pathlib.Path, str, dict[str, str]], None],
    on_task_complete: Callable[[Task, RunResult], list[Task]] | None = None,
    on_task_start: Callable[[Task, pathlib.Path, int], None] | None = None,
    max_heap_mb: int | None = None,
    current_threads: Callable[[], int] | None = None,
    max_dynamic_threads: int | None = None,
    task_priority: Callable[[Task, deque[Task]], tuple] | None = None,
) -> dict[tuple[str, str, str, int], RunResult]:
    results_map: dict[tuple[str, str, str, int], RunResult] = {}
    on_task_complete = on_task_complete or (lambda _task, _result: [])

    heap_budget = HeapBudget(max_heap_mb) if max_heap_mb is not None else None
    pending: dict[concurrent.futures.Future, tuple[tuple[str, str, str, int], Task]] = {}
    task_queue: deque[Task] = deque()

    def _active_tasks_for_priority() -> deque[Task]:
        active = deque(task_queue)
        for _future_key, task in pending.values():
            active.append(task)
        return active

    def _insert_task(next_task: Task) -> None:
        if task_priority is None:
            task_queue.append(next_task)
            return

        active = _active_tasks_for_priority()
        new_key = task_priority(next_task, active)
        for idx, queued_task in enumerate(task_queue):
            if new_key < task_priority(queued_task, active):
                task_queue.insert(idx, next_task)
                return
        task_queue.append(next_task)

    for task in initial_tasks:
        _insert_task(task)

    stop_scheduling = False

    dynamic_cap = max(1, int(max_dynamic_threads)) if max_dynamic_threads is not None else None
    executor_threads = max(1, int(threads))
    if current_threads is not None and dynamic_cap is not None:
        executor_threads = max(executor_threads, dynamic_cap)

    def _effective_threads() -> int:
        if current_threads is None:
            return executor_threads
        try:
            resolved = int(current_threads())
        except Exception:
            resolved = threads
        if dynamic_cap is not None:
            resolved = min(resolved, dynamic_cap)
        return max(1, resolved)

    def _submit_one(executor: concurrent.futures.ThreadPoolExecutor, task: Task) -> None:
        project_path = pathlib.Path(task.project).resolve()
        if heap_budget is not None:
            heap_budget.acquire(task.heap_mb)
        try:
            future = executor.submit(
                _run_with_retries,
                task,
                project_path,
                results_root,
                jvm_args,
                stagger_seconds,
                get_cli_args,
                execute_tests,
                commit_tests,
                on_task_start=on_task_start,
            )
        except Exception:
            if heap_budget is not None:
                heap_budget.release(task.heap_mb)
            raise
        key = (task.project, task.revision, task.model, task.run_number)
        pending[future] = (key, task)

    with concurrent.futures.ThreadPoolExecutor(max_workers=executor_threads) as executor:
        while task_queue or pending:
            target_threads = _effective_threads()
            if current_threads is None:
                pending_limit = max(1, target_threads * 2)
            else:
                pending_limit = max(1, target_threads)
            while not stop_scheduling and task_queue and len(pending) < pending_limit:
                _submit_one(executor, task_queue.popleft())

            if not pending:
                continue

            done, _not_done = concurrent.futures.wait(
                pending.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                key, finished_task = pending.pop(future)
                result = future.result()

                results_map[key] = result

                if heap_budget is not None:
                    heap_budget.release(finished_task.heap_mb)

                if result.outcome == RunOutcome.AGENT_ERROR:
                    stop_scheduling = True

                if not stop_scheduling:
                    for new_task in on_task_complete(finished_task, result):
                        _insert_task(new_task)

        if stop_scheduling:
            for future in list(pending.keys()):
                future.cancel()

    return results_map
