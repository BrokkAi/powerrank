#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import io
import sys
import random
import shutil
import tempfile
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.events import MouseDown, MouseScrollDown, MouseScrollUp
from textual.widgets import Footer, Header, ProgressBar, Static

import bpr
from bpr import RunOutcome, RunResult, Task, TaskKey


_BLOCK = "⣿"
_TILE_WIDTH = 35
_TILE_PADDING = 1
_TILE_HEIGHT = 6
_TILE_INNER_WIDTH = _TILE_WIDTH - (2 + _TILE_PADDING * 2)
_TICK_SECONDS = 0.5
_ETA_SAMPLE_WINDOW_SECONDS = 30 * 60.0
_ETA_MIN_SPAN_SECONDS = 15.0


class _TuiOutput(bpr._BprOutput):
    def __init__(self) -> None:
        super().__init__(stdout=lambda message: None, stderr=lambda message: print(message, file=sys.stderr))


class _ExitInterrupt(Exception):
    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(code)


class _ExitCapture:
    def __init__(self, on_exit: Callable[[int], None]) -> None:
        self._on_exit = on_exit

    def __call__(self, code: int) -> None:
        self._on_exit(code)
        raise _ExitInterrupt(code)


@dataclass
class _ProgressSnapshot:
    turn_requests: set[int]
    turn_log_titles: dict[int, str]
    turn_request_times: dict[int, float]


def _find_session_dir(history_root: Path) -> Path | None:
    session_dirs = sorted(
        (
            p
            for p in history_root.iterdir()
            if p.is_dir() and p.name[0:4].isdigit() and " Code " in p.name
        ),
        key=lambda p: p.name,
    )
    return session_dirs[-1] if session_dirs else None


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes = total // 60
    seconds_left = total % 60
    if minutes < 60:
        return f"{minutes}m {seconds_left}s"
    hours = minutes // 60
    minutes_left = minutes % 60
    return f"{hours}h {minutes_left}m"


def _format_tile_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds_left = divmod(total, 60)
    if minutes >= 10:
        return f"{minutes}m"
    return f"{minutes}:{seconds_left:02d}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return f"{text[: limit - 3]}..."


def _parse_task_progress(worktree: Path | None) -> _ProgressSnapshot:
    if worktree is None:
        return _ProgressSnapshot(set(), {}, {})

    history_root = worktree / ".brokk" / "llm-history"
    if not history_root.exists():
        return _ProgressSnapshot(set(), {}, {})

    session_dir = _find_session_dir(history_root)
    if session_dir is None:
        return _ProgressSnapshot(set(), {}, {})

    request_turns: set[int] = set()
    log_titles: dict[int, str] = {}
    request_times: dict[int, float] = {}

    for entry in sorted(session_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_file():
            continue

        name = entry.name
        parts = name.split(maxsplit=1)
        if len(parts) < 2:
            continue
        _, rest = parts
        if "-" not in rest:
            continue

        seq_text, suffix = rest.split("-", 1)
        if not seq_text.isdigit() or len(seq_text) != 3:
            continue
        seq = int(seq_text)

        if suffix == "request.json":
            request_turns.add(seq)
            try:
                request_times[seq] = entry.stat().st_mtime
            except OSError:
                request_times[seq] = time.time()
            continue

        if not suffix.endswith(".log"):
            continue

        title = suffix[:-4]
        if title.startswith(f"{seq_text}-"):
            title = title[len(seq_text) + 1 :]
        log_titles[seq] = title

    return _ProgressSnapshot(request_turns, log_titles, request_times)


def _safe_title_from_worktree(worktree: Path | None) -> str:
    if worktree is None:
        return ""

    history_root = worktree / ".brokk" / "llm-history"
    if not history_root.exists():
        return ""

    session_dir = _find_session_dir(history_root)
    if session_dir is None:
        return ""

    parts = session_dir.name.split(" ", 2)
    if len(parts) == 3:
        return f"{parts[1]} {parts[2]}"
    return session_dir.name


def _safe_path_exists(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.exists()
    except OSError:
        return False


def _zip_path_for_worktree(worktree: Path | None) -> Path | None:
    if worktree is None:
        return None
    return worktree.with_suffix(".zip")


def _worktree_replaced_with_zip(worktree: Path | None, *, was_present: bool) -> bool:
    if not was_present or worktree is None:
        return False
    if _safe_path_exists(worktree):
        return False
    return _safe_path_exists(_zip_path_for_worktree(worktree))


def _is_right_click(event: MouseDown) -> bool:
    button = getattr(event, "button", None)
    if button is None:
        return False
    if isinstance(button, int):
        return button in (2, 3)
    lowered = str(button).lower()
    return lowered in {"right", "button3", "mousebutton.right"}


class _TaskTile(Static):
    def __init__(
        self,
        task_key: TaskKey,
        panel: Panel,
    ) -> None:
        super().__init__(panel, classes="task-tile")
        self.styles.width = _TILE_WIDTH
        self.styles.min_width = _TILE_WIDTH
        self.styles.max_width = _TILE_WIDTH
        self.styles.height = _TILE_HEIGHT
        self.styles.min_height = _TILE_HEIGHT
        self.styles.max_height = _TILE_HEIGHT
        self.task_key = task_key

    def on_mouse_down(self, event: MouseDown) -> None:
        if not _is_right_click(event):
            return
        event.stop()
        app = self.app
        if isinstance(app, BPRTUI):
            app._copy_tile_worktree(self.task_key)


def _build_progress_text(
    snapshot: _ProgressSnapshot,
    *,
    task_started: bool,
) -> tuple[Text, str]:
    text = Text()
    plain_chunks: list[str] = []

    def _append(token: str, style: str) -> None:
        text.append(token, style=style)
        plain_chunks.append(token)

    seqs = sorted(set(snapshot.turn_requests) | set(snapshot.turn_log_titles))
    if not seqs:
        _append("preparing" if task_started else "queued", "dim")
        return text, "".join(plain_chunks)

    for seq in seqs:
        has_request = seq in snapshot.turn_requests
        has_log = seq in snapshot.turn_log_titles
        if has_request and has_log:
            _append(_BLOCK, "blue")
            continue
        if has_request:
            _append(_BLOCK, "yellow")
            break
        _append(_BLOCK, "yellow")
        break

    return text, "".join(plain_chunks)


def _build_parser() -> argparse.ArgumentParser:
    # Shared runner parser includes core options such as --filter.
    parser = bpr._build_parser()
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use fake in-process progress for testing without running workloads.",
    )
    parser.add_argument(
        "--mock-tasks",
        type=int,
        default=6,
        help="Number of mock tasks to simulate.",
    )
    return parser


class BPRTUI(App):
    CSS = """
    .overall-row {
        width: 100%;
        height: auto;
        margin: 0;
    }
    .overall-header-row {
        width: 100%;
        height: auto;
        margin: 0;
    }
    .overall-header {
        width: 1fr;
    }
    #overall-bar {
        width: 100%;
        min-width: 0;
    }
    #overall-bar > #bar {
        width: 1fr;
        min-width: 0;
    }
    #overall-eta {
        width: auto;
        min-width: 12;
        padding: 0 1;
        text-align: right;
    }
    .title-row {
        text-style: bold;
    }
    .task-tile {
        margin: 0 1;
        width: 35;
        min-width: 35;
        max-width: 35;
        height: 6;
        min-height: 6;
    }
    .task-row {
        height: 6;
        min-height: 6;
        max-height: 6;
    }
    .stats-row {
        width: 100%;
        height: auto;
    }
    .model-stats {
        width: 1fr;
        min-width: 0;
        height: auto;
        overflow-y: auto;
        padding: 0;
    }
    .project-stats {
        width: 1fr;
        min-width: 0;
        height: auto;
    }
    #tile-scroll {
        layout: vertical;
    }
    """

    BINDINGS = [("q", "quit", "Quit"), ("ctrl+q", "quit", "Quit"), ("meta+q", "quit", "Quit")]

    def __init__(
        self,
        parser_args: argparse.Namespace,
        unknown_args: list[str],
        *,
        mock: bool,
        mock_task_count: int,
    ) -> None:
        super().__init__()
        self._args = parser_args
        self._unknown_args = unknown_args
        self._mock = mock
        self._mock_task_count = max(1, mock_task_count)

        self._lock = threading.Lock()
        self._running = True
        self._done_event = threading.Event()
        self._exit_code = 0
        self._runner_stderr = ""

        self._task_states: dict[TaskKey, Task] = {}
        self._task_started_at: dict[TaskKey, float] = {}
        self._task_worktrees = bpr.TaskWorktreeMap()
        self._scheduled_by_model = Counter[str]()
        self._completed_by_model = Counter[str]()
        self._success_by_model = Counter[str]()
        self._duration_sum_by_model: dict[str, float] = {}
        self._timed_completed_by_model = Counter[str]()
        self._scheduled_by_project = Counter[str]()
        self._completed_by_project = Counter[str]()
        self._success_by_project = Counter[str]()
        self._duration_sum_by_project: dict[str, float] = {}
        self._timed_completed_by_project = Counter[str]()
        self._eta_samples = deque[tuple[float, int]]()
        self._eta_last_seconds: int | None = None
        self._eta_sample_window_seconds = _ETA_SAMPLE_WINDOW_SECONDS
        self._eta_min_span_seconds = _ETA_MIN_SPAN_SECONDS
        self._tile_widgets: dict[TaskKey, _TaskTile] = {}
        self._tile_keys: list[TaskKey] = []
        self._last_tiles_per_row = 0
        self._worktrees_seen_present: set[TaskKey] = set()
        self._runner_exception: BaseException | None = None

    def _results_root(self) -> Path:
        mode = getattr(self._args, "mode", "code") or "code"
        results_dir = getattr(self._args, "results_dir", None)
        return Path(results_dir) if results_dir else Path(f"{mode}results")

    def _history_seeded(self, completed_by_model: dict[str, int], success_by_model: dict[str, int]) -> None:
        with self._lock:
            for model, completed in completed_by_model.items():
                self._scheduled_by_model[model] += completed
                self._completed_by_model[model] += completed
            for model, success in success_by_model.items():
                self._success_by_model[model] += success
            models = set(self._scheduled_by_model.keys()) | set(self._success_by_model.keys())
            model_durations, model_timed_counts = self._seed_model_times(models=models)
            for model, total_seconds in model_durations.items():
                self._duration_sum_by_model[model] = self._duration_sum_by_model.get(model, 0.0) + total_seconds
            for model, timed_count in model_timed_counts.items():
                self._timed_completed_by_model[model] += timed_count
            project_totals, project_success, project_duration_sums, project_timed_counts = self._seed_project_times(
                models=models,
                runs=getattr(self._args, "runs", 1),
                rerun_mode=getattr(self._args, "rerun", "failed"),
            )
            for project, completed in project_totals.items():
                self._scheduled_by_project[project] += completed
                self._completed_by_project[project] += completed
            for project, success in project_success.items():
                self._success_by_project[project] += success
            for project, total_seconds in project_duration_sums.items():
                self._duration_sum_by_project[project] = (
                    self._duration_sum_by_project.get(project, 0.0) + total_seconds
                )
            for project, timed_count in project_timed_counts.items():
                self._timed_completed_by_project[project] += timed_count

    def _seed_model_times(
        self,
        *,
        models: set[str],
    ) -> tuple[dict[str, float], dict[str, int]]:
        duration_sum_by_model: dict[str, float] = {m: 0.0 for m in models}
        timed_count_by_model: dict[str, int] = {m: 0 for m in models}

        results_root = self._results_root()
        if not results_root.exists():
            return duration_sum_by_model, timed_count_by_model

        for entry in results_root.rglob("*.json"):
            if not entry.is_file():
                continue
            name = entry.name.removesuffix(".json")
            model, sep, _rest = name.rpartition("-")
            if not sep or model not in models:
                continue
            try:
                with open(entry, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except (json.JSONDecodeError, OSError):
                continue
            elapsed_millis = payload.get("elapsedMillis") if isinstance(payload, dict) else None
            if isinstance(elapsed_millis, int | float):
                duration_sum_by_model[model] += float(elapsed_millis) / 1000.0
                timed_count_by_model[model] += 1

        return duration_sum_by_model, timed_count_by_model

    def _selected_project_paths(self) -> set[str]:
        project = getattr(self._args, "project", None)
        if isinstance(project, str) and project:
            return {str(Path(project).resolve())}

        if "projects" not in vars(self._args):
            return set()

        project_spec = getattr(self._args, "projects", None)
        if project_spec is None:
            project_spec = ""
        dataset_names = [name.strip() for name in project_spec.split(",") if name.strip()]
        if not dataset_names:
            from dataset_config import DEFAULT_DATASETS

            dataset_names = list(DEFAULT_DATASETS)

        from dataset_config import get_dataset_config

        commits_dir = getattr(self._args, "commits_dir", "taskcommits-0126")
        selected_projects: set[str] = set()
        for dataset_name in dataset_names:
            try:
                config = get_dataset_config(dataset_name, commits_dir)
            except ValueError:
                continue
            selected_projects.add(str(Path(config.project_path).resolve()))
        return selected_projects

    def _resolve_project_for_run_dir(
        self,
        run_dir: Path,
        *,
        project_by_name: dict[str, str],
    ) -> str | None:
        for project_name, project_path in sorted(project_by_name.items(), key=lambda kv: len(kv[0]), reverse=True):
            if not run_dir.name.startswith(project_name):
                continue
            suffix = run_dir.name[len(project_name) :]
            if not suffix.isdigit():
                continue
            return project_path
        return None

    def _seed_project_times(
        self,
        *,
        models: set[str],
        runs: int,
        rerun_mode: str,
    ) -> tuple[
        dict[str, int],
        dict[str, int],
        dict[str, float],
        dict[str, int],
    ]:
        if not models:
            return {}, {}, {}, {}

        selected_projects = self._selected_project_paths()
        if not selected_projects:
            return {}, {}, {}, {}

        max_runs = max(1, runs)
        project_by_name = {Path(path).name: path for path in selected_projects}
        results_root = self._results_root()
        if not results_root.exists():
            return {}, {}, {}, {}

        outcomes_by_key: dict[tuple[str, str, str], dict[int, bpr.RunOutcome | None]] = {}
        elapsed_by_key: dict[tuple[str, str, str, int], float] = {}

        for run_dir in results_root.iterdir():
            if not run_dir.is_dir():
                continue

            project_path = self._resolve_project_for_run_dir(run_dir, project_by_name=project_by_name)
            if project_path is None:
                continue

            run_suffix = run_dir.name[len(Path(project_path).name) :]
            if not run_suffix.isdigit():
                continue

            run_number = int(run_suffix)
            if run_number > max_runs:
                continue

            for entry in run_dir.iterdir():
                if not entry.is_file():
                    continue
                name = entry.name
                if not name.endswith(".json"):
                    continue
                stem = entry.stem
                model, sep, revision = stem.rpartition("-")
                if not sep or model not in models:
                    continue
                try:
                    with open(entry, "r", encoding="utf-8") as fp:
                        payload = json.load(fp)
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(payload, dict):
                    continue
                stop_reason = str(payload.get("stopReason", ""))
                if stop_reason == "SUCCESS":
                    outcome = bpr.RunOutcome.SUCCESS
                elif stop_reason in ("HARNESS_TESTS_FAILED", "PREBUILD_FAILED"):
                    outcome = bpr.RunOutcome.TESTS_FAILED
                else:
                    outcome = bpr.RunOutcome.AGENT_FAILED

                outcomes_by_key.setdefault((project_path, model, revision), {})[run_number] = outcome
                elapsed_millis = payload.get("elapsedMillis")
                if isinstance(elapsed_millis, int | float):
                    elapsed_by_key[(project_path, model, revision, run_number)] = float(elapsed_millis) / 1000.0

        project_totals: dict[str, int] = {}
        project_success: dict[str, int] = {}
        project_duration_sum: dict[str, float] = {}
        project_timed_count: dict[str, int] = {}

        for (project_path, _model, _revision), by_run in outcomes_by_key.items():
            previous_outcome: bpr.RunOutcome | None = None
            for run_number in range(1, max_runs + 1):
                if not bpr._should_run_in_run(rerun_mode, run_number, previous_outcome):
                    break

                outcome = by_run.get(run_number)
                if outcome is None:
                    break

                project_totals[project_path] = project_totals.get(project_path, 0) + 1
                if outcome == bpr.RunOutcome.SUCCESS:
                    project_success[project_path] = project_success.get(project_path, 0) + 1

                elapsed_seconds = elapsed_by_key.get((project_path, _model, _revision, run_number))
                if elapsed_seconds is not None:
                    project_duration_sum[project_path] = project_duration_sum.get(project_path, 0.0) + elapsed_seconds
                    project_timed_count[project_path] = project_timed_count.get(project_path, 0) + 1

                previous_outcome = outcome

        return project_totals, project_success, project_duration_sum, project_timed_count

    def run_benchmark(self) -> None:
        if self._mock:
            self._run_mock()
            return

        exit_capture = _ExitCapture(self._handle_exit_code)
        try:
            bpr.run_with_args(
                self._args,
                self._unknown_args,
                output=_TuiOutput(),
                task_scheduled=self._task_scheduled,
                task_started=self._task_started,
                task_completed=self._task_completed,
                history_seeded=self._history_seeded,
                task_worktrees=self._task_worktrees,
                exit_fn=exit_capture,
            )
        except _ExitInterrupt:
            return

    def _run_benchmark_worker(self) -> None:
        stderr_capture = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr_capture):
                self.run_benchmark()
        except SystemExit as exc:
            self._runner_stderr = stderr_capture.getvalue().strip()
            self._exit_code = int(exc.code) if isinstance(exc.code, int) else 1
        except BaseException as exc:
            self._runner_stderr = stderr_capture.getvalue().strip()
            self._runner_exception = exc
            self._exit_code = 1
        finally:
            if not self._runner_stderr:
                self._runner_stderr = stderr_capture.getvalue().strip()
            self._done_event.set()

    def _handle_exit_code(self, code: int) -> None:
        self._exit_code = code
        self._done_event.set()

    def _task_scheduled(self, task: Task) -> None:
        key = bpr.task_key(task)
        with self._lock:
            self._task_states[key] = task
            self._task_worktrees.mark_scheduled(task)
            self._task_started_at.pop(key, None)
            self._worktrees_seen_present.discard(key)
            self._scheduled_by_model[task.model] += 1
            self._scheduled_by_project[task.project] += 1

    def _task_started(self, task: Task, worktree: Path, _attempt: int) -> None:
        key = bpr.task_key(task)
        with self._lock:
            if key in self._task_states:
                self._task_started_at[key] = time.time()
                self._worktrees_seen_present.discard(key)
                self._task_worktrees.mark_started(task, worktree)

    def _resolve_completed_outcome(self, task: Task) -> RunOutcome:
        result_path = bpr._results_json_path(
            self._results_root(),
            Path(task.project).name,
            task.run_number,
            task.model,
            task.revision,
        )
        outcome = bpr._read_run_outcome_from_results_file(result_path)
        return outcome if outcome is not None else RunOutcome.AGENT_FAILED

    def _task_completed_locked(self, task: Task, *, outcome: RunOutcome, sample_time: float) -> bool:
        key = bpr.task_key(task)
        if key not in self._task_states:
            return False
        self._completed_by_model[task.model] += 1
        self._completed_by_project[task.project] += 1
        if outcome == RunOutcome.SUCCESS:
            self._success_by_model[task.model] += 1
            self._success_by_project[task.project] += 1
        started_at = self._task_started_at.get(key)
        if started_at is not None:
            self._duration_sum_by_model[task.model] = self._duration_sum_by_model.get(task.model, 0.0) + (
                sample_time - started_at
            )
            self._timed_completed_by_model[task.model] += 1
            self._duration_sum_by_project[task.project] = self._duration_sum_by_project.get(task.project, 0.0) + (
                sample_time - started_at
            )
            self._timed_completed_by_project[task.project] += 1
        completed_total = sum(self._completed_by_model.values())
        self._record_eta_completion_sample(sample_time=sample_time, completed_total=completed_total)
        self._task_states.pop(key, None)
        self._task_started_at.pop(key, None)
        self._task_worktrees.mark_completed(task)
        self._worktrees_seen_present.discard(key)
        return True

    def _task_completed(self, task: Task, result: RunResult) -> None:
        now = time.time()
        with self._lock:
            self._task_completed_locked(task, outcome=result.outcome, sample_time=now)

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Header(show_clock=False, classes="overall-header"),
            Static("ETA --:--:--", id="overall-eta"),
            classes="overall-header-row",
        )
        yield Horizontal(
            # Textual ETA resets whenever `total` changes; we maintain a custom ETA label.
            ProgressBar(total=1, show_percentage=False, show_eta=False, id="overall-bar"),
            classes="overall-row",
        )
        yield Horizontal(
            ScrollableContainer(Static("No model stats yet", id="model-stats"), id="model-stats-scroll", classes="model-stats"),
            Static("No project stats yet", id="project-stats", classes="project-stats"),
            classes="stats-row",
        )
        yield Static("0 tasks queued", id="queued-count")
        yield ScrollableContainer(id="tile-scroll")
        yield Static("", id="status-flash")
        yield Footer()

    def on_mount(self) -> None:
        tile_scroll = self.query_one("#tile-scroll")
        tile_scroll.styles.layout = "vertical"
        self.call_later(self._refresh_loop)
        runner = threading.Thread(target=self._run_benchmark_worker, daemon=True)
        runner.start()

    def _refresh_loop(self) -> None:
        self._tick()
        if self._running:
            self.set_timer(_TICK_SECONDS, self._refresh_loop, name="refresh")

    def _should_exit_when_idle(
        self,
        *,
        has_active_tasks: bool,
        queued_total: int,
        scheduled_total: int,
    ) -> bool:
        # Avoid exiting on initial startup before any tasks are known.
        if has_active_tasks or queued_total != 0:
            return False
        return scheduled_total > 0 or self._done_event.is_set()

    def _build_task_panel(
        self,
        tasks: list[tuple[TaskKey, Task, Path | None, float | None]],
    ) -> list[tuple[TaskKey, Panel]]:
        now = time.time()
        tile_widgets: list[tuple[TaskKey, Panel]] = []

        for _key, task, worktree, started_at in tasks:
            row_title = _safe_title_from_worktree(worktree) or task.project
            snapshot = _parse_task_progress(worktree)
            if started_at is None:
                duration = "queued"
            else:
                duration = _format_tile_duration(now - started_at)

            rev_short = task.revision[:8]
            completed_turn = "No completed turn yet"
            if snapshot.turn_log_titles:
                latest = max(snapshot.turn_log_titles)
                completed_turn = snapshot.turn_log_titles[latest]

            active_turn_requests = sorted(set(snapshot.turn_requests) - set(snapshot.turn_log_titles))
            turn_duration = ""
            if active_turn_requests:
                current_turn = active_turn_requests[-1]
                turn_started = snapshot.turn_request_times.get(current_turn)
                if turn_started is None:
                    turn_duration = "n/a"
                else:
                    turn_duration = _format_tile_duration(now - turn_started)
            elif snapshot.turn_requests:
                # A future or completed turn exists, but there is no active turn yet.
                turn_duration = ""

            turn_text = _truncate(
                f"{rev_short} task {duration}"
                + (f" | turn {turn_duration}" if turn_duration else ""),
                _TILE_INNER_WIDTH,
            )

            progress_text, progress_plain = _build_progress_text(
                snapshot,
                task_started=started_at is not None,
            )
            lines = [
                turn_text,
                _truncate(row_title, _TILE_INNER_WIDTH),
                _truncate(completed_turn, _TILE_INNER_WIDTH),
                _truncate(progress_plain, _TILE_INNER_WIDTH),
            ]

            tile_text = Text()
            tile_text.append(lines[0])
            tile_text.append("\n")
            tile_text.append(lines[1])
            tile_text.append("\n")
            tile_text.append(lines[2])
            tile_text.append("\n")
            tile_text.append_text(progress_text)

            tile_panel = Panel(
                tile_text,
                title=f"{Path(task.project).name} · {task.model} · run {task.run_number}",
                border_style="blue",
                width=_TILE_WIDTH,
                height=_TILE_HEIGHT,
            )
            tile_widgets.append((_key, tile_panel))

        return tile_widgets

    def _refresh_tiles(
        self,
        tile_scroll: ScrollableContainer,
        panels: list[tuple[TaskKey, Panel]],
        tiles_per_row: int,
    ) -> None:
        current_keys = [key for key, _panel in panels]
        layout_changed = (
            current_keys != self._tile_keys
            or tiles_per_row != self._last_tiles_per_row
            or any(key not in self._tile_widgets for key in current_keys)
            or len(current_keys) != len(self._tile_widgets)
        )
        if layout_changed:
            tile_scroll.remove_children()
            self._tile_widgets = {}
            for key, panel in panels:
                widget = _TaskTile(key, panel)
                self._tile_widgets[key] = widget
            for start in range(0, len(panels), tiles_per_row):
                row = Horizontal(
                    *[self._tile_widgets[key] for key, _ in panels[start : start + tiles_per_row]],
                    classes="task-row",
                )
                row.styles.height = _TILE_HEIGHT
                row.styles.min_height = _TILE_HEIGHT
                row.styles.max_height = _TILE_HEIGHT
                tile_scroll.mount(row)
        else:
            for key, panel in panels:
                widget = self._tile_widgets.get(key)
                if widget is None:
                    layout_changed = True
                    break
                widget.update(panel)
            if layout_changed:
                self._refresh_tiles(tile_scroll, panels, tiles_per_row)
                return

        self._tile_keys = current_keys
        self._last_tiles_per_row = tiles_per_row

    def _collect_active_task_items_locked(
        self,
        *,
        sample_time: float,
    ) -> list[tuple[TaskKey, Task, Path | None, float | None]]:
        task_items: list[tuple[TaskKey, Task, Path | None, float | None]] = []
        for key, task in list(self._task_states.items()):
            started_at = self._task_started_at.get(key)
            if started_at is None:
                continue
            worktree = self._task_worktrees.get_by_key(key)
            if _safe_path_exists(worktree):
                self._worktrees_seen_present.add(key)
            elif _worktree_replaced_with_zip(worktree, was_present=key in self._worktrees_seen_present):
                # Reconcile missed worker completion callbacks based on archive handoff.
                outcome = self._resolve_completed_outcome(task)
                self._task_completed_locked(task, outcome=outcome, sample_time=sample_time)
                continue
            task_items.append((key, task, worktree, started_at))
        return task_items

    def _tick(self) -> None:
        with self._lock:
            now = time.time()
            task_items = self._collect_active_task_items_locked(sample_time=now)

            rerun_mode = getattr(self._args, "rerun", "failed")
            max_runs = max(1, int(getattr(self._args, "runs", 1)))
            models_that_might_get_more_tasks: dict[str, bool] = {}
            projects_that_might_get_more_tasks: dict[str, bool] = {}
            for _key, task in self._task_states.items():
                if task.run_number >= max_runs:
                    continue
                if rerun_mode not in {"failed", "all"}:
                    continue
                models_that_might_get_more_tasks[task.model] = True
                projects_that_might_get_more_tasks[task.project] = True

            scheduled_total = sum(self._scheduled_by_model.values())
            completed_total = sum(self._completed_by_model.values())
            model_totals = dict(self._scheduled_by_model)
            model_completed = dict(self._completed_by_model)
            model_success = dict(self._success_by_model)
            project_totals = dict(self._scheduled_by_project)
            project_completed = dict(self._completed_by_project)
            project_success = dict(self._success_by_project)
            eta_seconds = self._estimate_eta_seconds(
                now=now,
                scheduled_total=scheduled_total,
                completed_total=completed_total,
            )

        self.sub_title = f"{completed_total}/{scheduled_total}"
        self.query_one("#overall-eta").update(self._format_eta_label(eta_seconds))

        overall_bar = self.query_one("#overall-bar")
        if scheduled_total:
            overall_bar.update(total=scheduled_total, progress=completed_total)
        else:
            overall_bar.update(total=1, progress=0)

        model_rows: list[tuple[str, str, str, str, str]] = []
        model_pass_rates: dict[str, float] = {}
        for model in sorted(set(model_totals) | set(model_completed)):
            total = model_totals.get(model, 0)
            done = model_completed.get(model, 0)
            success = model_success.get(model, 0)
            remaining = max(0, total - done)
            fail = max(0, done - success)
            timed_count = self._timed_completed_by_model.get(model, 0)
            model_pass_rates[model] = success / done if done else 0.0
            if timed_count:
                time_label = _format_duration(self._duration_sum_by_model.get(model, 0.0) / timed_count)
            else:
                time_label = "n/a"
            remaining_text = str(remaining)
            if models_that_might_get_more_tasks.get(model):
                remaining_text = f"{remaining_text}+"
            model_rows.append((model, str(success), str(fail), remaining_text, time_label))

        model_rows = sorted(model_rows, key=lambda row: model_pass_rates[row[0]], reverse=True)

        project_rows: list[tuple[str, str, str, str, str, str]] = []
        project_pass_rates: dict[str, float] = {}
        for project in sorted(set(project_totals) | set(project_completed)):
            total = project_totals.get(project, 0)
            done = project_completed.get(project, 0)
            success = project_success.get(project, 0)
            remaining = max(0, total - done)
            fail = max(0, done - success)
            timed_count = self._timed_completed_by_project.get(project, 0)
            project_pass_rates[project] = success / done if done else 0.0
            project_label = Path(project).name
            if timed_count:
                time_label = _format_duration(self._duration_sum_by_project.get(project, 0.0) / timed_count)
            else:
                time_label = "n/a"
            remaining_text = str(remaining)
            if projects_that_might_get_more_tasks.get(project):
                remaining_text = f"{remaining_text}+"
            project_rows.append((project, project_label, str(success), str(fail), remaining_text, time_label))

        project_rows = sorted(project_rows, key=lambda row: project_pass_rates[row[0]], reverse=True)

        if not model_rows:
            self.query_one("#model-stats").update("No model stats yet")
        else:
            table = Table(
                show_header=True,
                show_lines=False,
                box=box.SIMPLE,
                padding=(0, 1),
                expand=False,
                show_edge=True,
                row_styles=["", "#bfbfbf"],
            )
            table.add_column("Model", no_wrap=True)
            table.add_column("Pass", justify="right", no_wrap=True)
            table.add_column("Fail", justify="right", no_wrap=True)
            table.add_column("Remaining", justify="right", no_wrap=True)
            table.add_column("Time per Task", justify="right", no_wrap=True)
            for model_row in model_rows:
                table.add_row(*model_row)
            self.query_one("#model-stats").update(table)
        model_scroll = self.query_one("#model-stats-scroll")
        model_scroll_height = 6
        model_scroll.styles.height = model_scroll_height
        model_scroll.styles.max_height = model_scroll_height
        model_scroll.styles.min_height = model_scroll_height

        if not project_rows:
            self.query_one("#project-stats").update("No project stats yet")
        else:
            table = Table(
                show_header=True,
                show_lines=False,
                box=box.SIMPLE,
                padding=(0, 1),
                expand=False,
                show_edge=True,
                row_styles=["", "#bfbfbf"],
            )
            table.add_column("Project", no_wrap=True)
            table.add_column("Pass", justify="right", no_wrap=True)
            table.add_column("Fail", justify="right", no_wrap=True)
            table.add_column("Remaining", justify="right", no_wrap=True)
            table.add_column("Time per Task", justify="right", no_wrap=True)
            for project_row in project_rows:
                table.add_row(project_row[1], project_row[2], project_row[3], project_row[4], project_row[5])
            self.query_one("#project-stats").update(table)
        queued_total = max(0, scheduled_total - completed_total - len(task_items))
        self.query_one("#queued-count").update(f"{queued_total} tasks queued")

        if self._should_exit_when_idle(
            has_active_tasks=bool(task_items),
            queued_total=queued_total,
            scheduled_total=scheduled_total,
        ):
            self._running = False
            self.exit(self._exit_code)
            return

        tile_scroll = self.query_one("#tile-scroll")
        if not task_items:
            if self._done_event.is_set():
                self._running = False
            tile_scroll.remove_children()
            self._tile_widgets = {}
            self._tile_keys = []
            self._last_tiles_per_row = 0
            tile_scroll.mount(Static("No active tasks"))
            return

        task_items = self._ordered_active_task_items(task_items)
        tiles = self._build_task_panel(task_items)
        if not tiles:
            tile_scroll.remove_children()
            tile_scroll.mount(Static("No active tasks"))
            self._tile_widgets = {}
            self._tile_keys = []
            self._last_tiles_per_row = 0
            return

        available_width = max(1, self.size.width - 4)
        tile_stride = max(1, _TILE_WIDTH + (_TILE_PADDING * 2))
        tiles_per_row = max(1, available_width // tile_stride)

        self._refresh_tiles(tile_scroll, tiles, tiles_per_row)

    def _ordered_active_task_items(
        self,
        task_items: list[tuple[TaskKey, Task, Path | None, float | None]],
    ) -> list[tuple[TaskKey, Task, Path | None, float | None]]:
        return sorted(task_items, key=lambda entry: (entry[3] if entry[3] is not None else float("inf"), entry[0]))

    def _record_eta_completion_sample(self, *, sample_time: float, completed_total: int) -> None:
        self._eta_samples.append((sample_time, completed_total))
        cutoff = sample_time - self._eta_sample_window_seconds
        while len(self._eta_samples) > 1 and self._eta_samples[0][0] < cutoff:
            self._eta_samples.popleft()

    def _estimate_eta_seconds(self, *, now: float, scheduled_total: int, completed_total: int) -> int | None:
        remaining = scheduled_total - completed_total
        if remaining <= 0:
            self._eta_last_seconds = 0
            return 0

        cutoff = now - self._eta_sample_window_seconds
        while len(self._eta_samples) > 1 and self._eta_samples[0][0] < cutoff:
            self._eta_samples.popleft()

        if len(self._eta_samples) < 2:
            return self._eta_last_seconds

        start_time, start_completed = self._eta_samples[0]
        end_time, end_completed = self._eta_samples[-1]
        delta_seconds = end_time - start_time
        delta_completed = end_completed - start_completed
        if delta_completed <= 0 or delta_seconds < self._eta_min_span_seconds:
            return self._eta_last_seconds

        completion_rate = delta_completed / delta_seconds
        eta_seconds = max(1, ceil(remaining / completion_rate))
        self._eta_last_seconds = eta_seconds
        return eta_seconds

    def _format_eta_label(self, eta_seconds: int | None) -> str:
        if eta_seconds is None:
            return "ETA --:--:--"

        total = max(0, eta_seconds)
        minutes, seconds = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 999999:
            return "ETA +999999h"
        return f"ETA {hours:02}:{minutes:02}:{seconds:02}"

    def _copy_tile_worktree(self, task_key: TaskKey) -> None:
        worktree = self._task_worktrees.get_by_key(task_key)
        if worktree is None:
            self._flash_status(f"No worktree for {task_key[0]}/{task_key[1][:8]}")
            return
        try:
            self.copy_to_clipboard(str(worktree))
            self._flash_status(f"Copied worktree: {worktree}")
        except Exception:
            self._flash_status(f"Failed to copy: {worktree}")

    def _flash_status(self, message: str) -> None:
        self._set_status_message(message)
        self.set_timer(1.5, self._clear_status, name="status-flash-clear")

    def _set_status_message(self, message: str) -> None:
        self.query_one("#status-flash").update(message)

    def _clear_status(self) -> None:
        self.query_one("#status-flash").update("")

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self._scroll_tile_view(-1)
        event.stop()

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self._scroll_tile_view(1)
        event.stop()

    def _scroll_tile_view(self, direction: int) -> None:
        tile_scroll = self.query_one("#tile-scroll")
        try:
            if direction < 0:
                tile_scroll.scroll_up(distance=4)
            else:
                tile_scroll.scroll_down(distance=4)
        except TypeError:
            if direction < 0:
                tile_scroll.scroll_up()
            else:
                tile_scroll.scroll_down()

    def _run_mock(self) -> None:
        tmp_root = Path(tempfile.mkdtemp(prefix="bpr-tui-mock-"))
        for idx in range(self._mock_task_count):
            if not self._running:
                break

            revision = f"{idx:07x}deadbeef"
            model = random.choice(["mock-small", "mock-large"])
            task = Task(
                project=f"{tmp_root / 'project'}",
                revision=revision,
                model=model,
                run_number=1,
                job_env=None,
                heap_mb=1024,
                properties={},
            )
            self._task_scheduled(task)

            worktree = tmp_root / f"mock-task-{idx}"
            session_root = (
                worktree
                / ".brokk"
                / "llm-history"
                / f"2026-02-19-16-38-14 Code Task {idx}"
            )
            session_root.mkdir(parents=True, exist_ok=True)
            self._task_started(task, worktree, 1)

            time.sleep(random.uniform(0.4, 0.8))

            for turn in range(1, 3):
                ts = time.strftime("%H-%M.%S")
                request_file = session_root / f"{ts} {turn:03d}-request.json"
                request_file.write_text("{}", encoding="utf-8")
                if not self._running:
                    break
                time.sleep(random.uniform(0.4, 0.8))
                log_file = session_root / f"{ts} {turn:03d}-Mock turn {turn}.log"
                log_file.write_text(f"Mock turn {turn}", encoding="utf-8")
                if not self._running:
                    break

            if self._running:
                self._task_completed(task, RunResult(outcome=RunOutcome.SUCCESS))
        self._handle_exit_code(0)
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    parser = _build_parser()
    args, unknown = parser.parse_known_args()
    if args.mode == "infer-context":
        parser.error("--infer-context is not supported by bpr_tui.py")

    app = BPRTUI(
        args,
        unknown,
        mock=args.mock,
        mock_task_count=args.mock_tasks,
    )
    app.run()
    if app._runner_stderr and (app._runner_exception is not None or app._exit_code != 0):
        print(app._runner_stderr, file=sys.stderr)
    if app._runner_exception is not None:
        raise RuntimeError("BPR TUI benchmark worker crashed") from app._runner_exception


if __name__ == "__main__":
    main()
