import argparse
import datetime
import enum
import json
import os
import pathlib
import subprocess
import sys
import shutil
import concurrent.futures
import archive
from typing import Dict, List
import random
import re
import time
import tempfile
from collections import defaultdict

try:
    import fcntl  # POSIX advisory locking
except ImportError:  # pragma: no cover
    fcntl = None  # Not available on some platforms (e.g., Windows)

import pystache

from scan_commits import has_test_word


class RunResult(enum.Enum):
    SUCCESS = 0
    AGENT_FAILED = 1
    TESTS_FAILED = 2


CLI_BIN = pathlib.Path("../brokk/cli")


def _run_cli(cmd: List[str], log_file: pathlib.Path) -> subprocess.CompletedProcess:
    """
    Helper to execute Brokk CLI commands and append output
    to the supplied log-file. If the BB_DEBUG environment variable
    is set, the output is also echoed to the console.
    """
    # The full command (including the CLI binary) is now supplied by the caller.
    full_cmd = cmd

    # If debugging, show the command being executed.
    if os.getenv("BB_DEBUG"):
        print(f"Running command: {' '.join(full_cmd)}", file=sys.stderr)

    # Launch the subprocess and stream combined stdout/stderr so that we
    # can forward each chunk immediately.
    with subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # default buffering; avoids RuntimeWarning for line buffering in binary mode
    ) as proc, open(log_file, "ab") as log_fp:
        # Stream child process output line-by-line to minimise latency.
        for line in proc.stdout:
            log_fp.write(line)
            if os.getenv("BB_DEBUG"):
                try:
                    sys.stderr.buffer.write(line)
                    sys.stderr.flush()
                except AttributeError:
                    # Fallback if sys.stderr lacks 'buffer'.
                    sys.stderr.write(line.decode(errors="replace"))
                    sys.stderr.flush()
        proc.wait()

    # Return a CompletedProcess to maintain the original API contract.
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=None, stderr=None)


def run_one_revision(project: str, revision: str, model: str, mode: str, cli_suffix_part: str, run_number: int, jvm_args: List[str], stagger: bool, exclude: List[str]) -> RunResult:
    project_path = pathlib.Path(project).resolve()
    if stagger:
        time.sleep(random.uniform(0, 2))
    if not (project_path / ".git").is_dir():
        raise ValueError(f"Project '{project}' is not a git repository.")

    # ------------------------------------------------------------------
    # 1. set up the worktree
    # ------------------------------------------------------------------
    def _git_generic(root, *git_args: str) -> subprocess.CompletedProcess[str]:
        """
        Helper to run git commands inside the work-tree and return the result.
        Raises CalledProcessError if the command fails.
        """
        try:
            return subprocess.run(
                ["git", *git_args],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"""
            -----
            Error executing command: {e.cmd} in {root}
            {e.output}
            -----
            """)
            raise e

    # Resolve commit hashes for the target revision and its parent
    revshort = _git_generic(project_path, "rev-parse", "--short", revision)

    session = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    workdir_name = f"{model}-{revshort}-{run_number}-{session}"
    worktree_path = pathlib.Path.home() / "brokkbench" / project_path.name / workdir_name
    try:
        os.makedirs(worktree_path.parent, exist_ok=True)

        # worktree git-runner now that we've defined it
        def _git(*args, **kwargs):
            return _git_generic(worktree_path, *args, **kwargs)

        agent_log_path = worktree_path.parent / f"{workdir_name}-agent.txt"
        run_output_path = worktree_path / "run-output.txt"

        # ------------------------------------------------------------------
        # Load revision-specific properties
        # ------------------------------------------------------------------
        props_dir = pathlib.Path("codetasks/")
        props_path = props_dir / f"{revision}.properties"

        # Defaults if no properties file is present
        props: dict[str, str] = {}
        props_extra_tests: list[str] = []
        props_testall_cmd: str | None = None

        if props_path.exists():
            # --------------------------------------------------------------
            # Load revision-specific properties
            # --------------------------------------------------------------
            with open(props_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    props[key.strip()] = value.strip()

            extra_tests_raw: str = props.get("extra_tests", "")
            props_extra_tests = [p.strip() for p in extra_tests_raw.split(",") if p.strip()]
            props_testall_cmd = props.get("testall_cmd")

            # Ensure the properties file provides at least one required key
            if not props_extra_tests and not props_testall_cmd:
                raise ValueError(
                    f"Properties file '{props_path}' must specify at least one "
                    "of 'extra_tests' or 'testall_cmd'."
                )

        def _write_run_output(paths: List[pathlib.Path]) -> None:
            """
            Combine the supplied log files into `run-output.txt`
            inside the worktree directory. Missing paths are ignored.
            """
            worktree_path.mkdir(parents=True, exist_ok=True)
            with open(run_output_path, "wb") as out_fp:
                for p in paths:
                    if p is None or not p.exists():
                        continue
                    out_fp.write(f"-------------- {p}\n".encode())
                    out_fp.flush()
                    with open(p, "rb") as src_fp:
                        shutil.copyfileobj(src_fp, out_fp)
                    out_fp.write(b"\n")

        # Create work-tree via Brokk CLI
        first_cmd = [
            str(CLI_BIN),
            *jvm_args,
            f"--project={project_path}",
            f"--worktree={worktree_path}",
        ]
        if _run_cli(first_cmd, agent_log_path).returncode != 0:
            _write_run_output([agent_log_path])
            return RunResult.AGENT_FAILED

        # --------------------------------------------------------------
        # Determine repository name (from origin URL) for build-lock key
        # --------------------------------------------------------------
        origin_url = _git("config", "--get", "remote.origin.url")
        # Extract final path component (handles '/' or ':' separators) and
        # strip a trailing '.git' if present.
        last_part = origin_url.split("/")[-1].split(":")[-1]
        origin_repo_name = re.sub(r"\.git$", "", last_part)

        # Reset work-tree to the parent of the target revision
        _git("reset", "--hard", f"{revision}^")

        # Identify all files touched by the target revision
        diff_output = _git(
            "diff", "--name-status", "--no-renames", f"{revision}^", revision
        ).splitlines()

        edit_files: List[str] = []
        test_files: List[str] = []

        for line in diff_output:
            if not line.strip():
                continue
            status, path = line.split("\t", 1)
            path = path.strip()

            if has_test_word(path):
                if status == "D":
                    # File was deleted in the target revision – remove it and
                    # do NOT include it in the list of tests to read.
                    _git("rm", "--", path)
                    continue
                test_files.append(path)
                # Checkout the file from the target revision into the work-tree
                _git("checkout", revision, "--", path)
                _git("add", "--", path)
            else:
                edit_files.append(path)

        # ------------------------------------------------------------------
        # Add any extra tests specified in the properties file
        # ------------------------------------------------------------------
        for et in props_extra_tests:
            if et not in test_files:
                test_files.append(et)
                try:
                    _git("checkout", revision, "--", et)
                except subprocess.CalledProcessError:
                    raise ValueError(
                        f"Extra test file '{et}' specified in properties for revision "
                        f"'{revision}' does not exist."
                    )
                _git("add", "--", et)

        # Ensure that all edit files exist in the work-tree.  Any file that is
        # part of the edited list but absent in the current work-tree (for example,
        # it is newly added in the target revision) is created as an empty file so
        # the agent has something to modify.
        for f in edit_files:
            abs_path = worktree_path / f
            if not abs_path.exists():
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.touch()
                _git("add", "--", f)

        # Validate that at least one test file exists.
        if not test_files:
            raise ValueError(
                f"No test files detected for revision '{revision}' "
                "and none specified via 'extra_tests'."
            )

        # Commit the test-only snapshot (should not be empty
        _git("commit", "-m", "BrokkBench: extract test files")



        # ------------------------------------------------------------------
        # 3. Run Brokk CLI agent task
        # ------------------------------------------------------------------
        cli_bin_for_task = os.getenv("BRKBCH_CLI", str(CLI_BIN))
        second_cmd: List[str] = [
            cli_bin_for_task,
            *jvm_args,
            f"--project={project_path}",
            f"--worktree={worktree_path}",
            "--deepscan",
        ]
        for f in edit_files:
            # Skip any file that matches one of the exclude patterns
            if any(pathlib.Path(f).match(p) for p in exclude):
                continue
            second_cmd.append(f"--edit={f}")
        for f in test_files:
            # Only pass files that still exist (i.e. were not deleted)
            if (worktree_path / f).exists():
                second_cmd.append(f"--read={f}")

        # Pass model parameter to agent
        if model:
            second_cmd.append(f"--model={model}")

        code_spec = f"@codetasks/{revision}.txt"
        second_cmd.append(f"--code={code_spec}")

        if _run_cli(second_cmd, agent_log_path).returncode != 0:
            _write_run_output([agent_log_path])
            return RunResult.AGENT_FAILED

        # ------------------------------------------------------------------
        # 3b. Extract metrics from agent log and write to results file
        # ------------------------------------------------------------------
        metrics_json: str | None = None
        with open(agent_log_path, "r", encoding="utf-8") as log_fp:
            for line in log_fp:
                if line.startswith("BRK_CODEAGENT_METRICS="):
                    metrics_json = line.strip().split("=", 1)[1]
        if metrics_json is None:
            raise ValueError("Metrics not found in output!")
        metrics = json.loads(metrics_json)
        metrics['worktree'] = str(worktree_path)

        # ------------------------------------------------------------------
        # 4. Commit agent's work
        # ------------------------------------------------------------------
        _git("add", "-A")
        _git("commit", "--allow-empty", "-m", "Agent work")

        # ------------------------------------------------------------------
        # 4b. Evaluate stop reason and prepare results path
        # ------------------------------------------------------------------
        results_dir = pathlib.Path(f"{mode}results{cli_suffix_part}") / f"{project_path.name}{run_number}"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"{model}-{revision}.json"

        if metrics.get("stopReason") != "SUCCESS":
            with open(results_path, "w", encoding="utf-8") as res_fp:
                json.dump(metrics, res_fp)
                res_fp.write("\n")
            _write_run_output([agent_log_path])
            return RunResult.TESTS_FAILED

        # ------------------------------------------------------------------
        # 5. Run test harness
        # ------------------------------------------------------------------
        test_log_path = worktree_path / "tests.txt"
        with open(test_log_path, "wb") as log:
            # Determine the test command (property overrides env).
            test_command = props_testall_cmd or os.getenv("BRK_TESTALL_CMD")
            if not test_command:
                log.write("No TESTALL cmd found, deriving from TESTSOME\n".encode())
                # Fallback to template interpolation
                env_template = os.getenv("BRK_TESTSOME_CMD")
                if env_template:
                    template = env_template
                else:
                    properties_file = project_path / ".brokk" / "project.properties"
                    build_details_json_str: str | None = None
                    if not properties_file.exists():
                        raise ValueError(
                            f"No test command specified. Tried 'testall_cmd' property, "
                            f"BRK_TESTALL_CMD env var, BRK_TESTSOME_CMD env var, and looked "
                            f"for {properties_file} which was not found."
                        )

                    with open(properties_file, "r", encoding="utf-8") as fp:
                        for line in fp:
                            stripped = line.strip()
                            if not stripped or stripped.startswith("#"):
                                continue
                            if stripped.startswith("buildDetailsJson="):
                                build_details_json_str = stripped.split("=", 1)[1].strip()
                                build_details_json_str = build_details_json_str.replace(r"\:", ":")
                                break

                    if build_details_json_str is None:
                        raise ValueError("'buildDetailsJson' not found in project.properties")

                    build_details = json.loads(build_details_json_str)
                    template = build_details.get("testSomeCommand")
                    if not template:
                        raise ValueError("'testSomeCommand' not found in build details JSON")

                context: Dict[str, List[str]] = {}
                if "{{#classes}}" in template:
                    if "{{#files}}" in template:
                        raise ValueError("Template cannot contain both #classes and #files")
                    _class_values = [pathlib.Path(p).stem for p in test_files if p.endswith(".java")]
                    context["classes"] = [
                            {"value": v, "first": i == 0, "last": i == len(_class_values) - 1}
                            for i, v in enumerate(_class_values)
                        ]
                elif "{{#files}}" in template:
                    context["files"] = [
                            {"value": f, "first": i == 0, "last": i == len(test_files) - 1}
                            for i, f in enumerate(test_files)
                        ]
                else:
                    raise ValueError("Template must contain either #classes or #files")

                log.write(f"Harness test template is {template} {context}\n".encode())
                test_command = pystache.render(template, context)

            log.write(("Running " + test_command + "\n").encode())

            # --------------------------------------------------------------
            # Optionally enforce single-build execution via file lock
            # --------------------------------------------------------------
            no_concurrent_builds = os.getenv("BRK_NO_CONCURRENT_BUILDS", "").lower() == "true"

            def _run_verification() -> subprocess.CompletedProcess:
                return subprocess.run(
                    test_command,
                    shell=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=worktree_path,
                )

            if not no_concurrent_builds:
                # Concurrency allowed – run immediately
                test_proc = _run_verification()
            else:
                lock_dir = pathlib.Path(tempfile.gettempdir()) / "brokk"
                try:
                    lock_dir.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    log.write(f"Unable to create lock directory {lock_dir}; proceeding without build lock: {exc}\n".encode())
                    log.flush()
                    test_proc = _run_verification()
                else:
                    lock_file = lock_dir / f"{origin_repo_name}.lock"
                    try:
                        lock_fp = open(lock_file, "w")
                    except Exception as exc:
                        log.write(f"Failed to open lock file {lock_file}; proceeding without build lock: {exc}\n".encode())
                        log.flush()
                        test_proc = _run_verification()
                    else:
                        try:
                            if fcntl is not None:
                                fcntl.flock(lock_fp, fcntl.LOCK_EX)
                                log.write(f"Acquired build lock {lock_file}\n".encode())
                                log.flush()
                                test_proc = _run_verification()
                            else:
                                log.write(f"fcntl not available; proceeding without build lock {lock_file}\n".encode())
                                log.flush()
                                test_proc = _run_verification()
                        finally:
                            try:
                                lock_fp.close()
                            except Exception:
                                pass

        # Override the agent's self-reported success if tests failed
        tests_failed = test_proc.returncode != 0
        if tests_failed:
            metrics["stopReason"] = "HARNESS_TESTS_FAILED"

        # Persist metrics (updated if necessary)
        with open(results_path, "w", encoding="utf-8") as res_fp:
            json.dump(metrics, res_fp)
            res_fp.write("\n")

        _write_run_output([agent_log_path, test_log_path])

        if tests_failed:
            return RunResult.TESTS_FAILED
        return RunResult.SUCCESS
    finally:
        archive.archive_worktree(project_path, worktree_path)


def run_with_retries(project: str,
                     revision: str,
                     model: str,
                     mode: str,
                     cli_suffix_part: str,
                     run_number: int,
                     jvm_args: List[str],
                     stagger: bool,
                     exclude: List[str]) -> RunResult:
    """
    Keep invoking `run_one_revision` until the agent's metrics JSON
    no longer contains a `stopExplanation` that includes
    the phrase ``check litellm logs`` (case-insensitive).

    Returns the final RunResult from the last invocation.
    """
    project_path = pathlib.Path(project).resolve()

    while True:
        result = run_one_revision(
            project, revision, model, mode,
            cli_suffix_part, run_number, jvm_args, stagger, exclude
        )

        # Locate the metrics file written by `run_one_revision`.
        results_dir = pathlib.Path(f"{mode}results{cli_suffix_part}") / f"{project_path.name}{run_number}"
        results_path = results_dir / f"{model}-{revision}.json"

        if not results_path.exists():
            # No metrics produced; give up.
            return result

        try:
            with open(results_path, "r", encoding="utf-8") as fp:
                metrics = json.load(fp)
        except Exception:
            # Failed to load metrics; stop retrying.
            return result

        stop_expl = str(metrics.get("stopExplanation", "")).lower()
        if "check litellm logs" in stop_expl or "RateLimitError" in stop_expl:
            # Retry
            continue

        # Explanation is acceptable – stop retrying.
        return result


def main() -> None:
    os.environ["BRK_CODEAGENT_METRICS"] = "true"

    parser = argparse.ArgumentParser(description="Run Brokk agent harness.")
    parser.add_argument("--project", required=True, help="Git project directory.")
    parser.add_argument(
        "--model",
        required=True,
        help="Comma-separated list of model names to pass to the agent.",
    )
    parser.add_argument("--mode", required=True, help="Mode prefix for results directory (e.g., 'train').")
    parser.add_argument(
        "--threads",
        type=int,
        default=20,
        help="Number of parallel threads to process revisions.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run each revision/model combination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show task counts and exit without executing runs.",
    )
    parser.add_argument(
        "--stagger",
        action="store_true",
        help="Add a random 0-2s sleep before starting each task.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Filenames or glob patterns to exclude from --edit options. "
             "May be supplied multiple times.",
    )
    args, unknown_args = parser.parse_known_args()
    # Forward JVM-style flags to the CLI; error on anything else.
    jvm_args: List[str] = [ua for ua in unknown_args if ua.startswith("-X") or ua.startswith("-D")]
    invalid_unknowns = [ua for ua in unknown_args if ua not in jvm_args]
    if invalid_unknowns:
        print(f"Unknown arguments: {' '.join(invalid_unknowns)}", file=sys.stderr)
        sys.exit(5)

    project_path = pathlib.Path(args.project).resolve()

    # ------------------------------------------------------------------
    # Read revisions (first token per non-empty line) from stdin
    # ------------------------------------------------------------------
    revisions: List[str] = []
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        revisions.append(stripped.split()[0])

    # ------------------------------------------------------------------
    # Cross-reference requested revisions with existing task files
    # ------------------------------------------------------------------
    tasks_dir = pathlib.Path("codetasks/")
    existing_revisions = [rev for rev in revisions if (tasks_dir / f"{rev}.txt").exists()]
    if len(revisions) != len(existing_revisions):
        _missing = [rev for rev in revisions if rev not in existing_revisions]
        print(
            f"Warning: {len(revisions)} revisions requested, but only "
            f"{len(existing_revisions)} tasks found on disk",
            file=sys.stderr,
        )
    # Continue processing only those revisions that have corresponding task files.
    revisions = existing_revisions

    if not revisions:
        print("No valid revisions with task files provided on stdin.", file=sys.stderr)
        sys.exit(3)

    # ------------------------------------------------------------------
    # Validate revisions before doing any work
    # ------------------------------------------------------------------

    invalid_revs: List[str] = []
    for rev in revisions:
        # Ensure the revision exists
        try:
            subprocess.run(
                ["git", "-C", str(project_path), "rev-parse", "--verify", f"{rev}^{{commit}}"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            invalid_revs.append(rev)
            continue

        # Ensure the revision has exactly one parent (no merge commits)
        parent_line = subprocess.check_output(
            ["git", "-C", str(project_path), "rev-list", "--parents", "-n", "1", rev],
            text=True,
        ).strip()
        parent_count = len(parent_line.split()) - 1  # first token is the commit itself
        if parent_count != 1:
            invalid_revs.append(rev)

    if invalid_revs:
        print("Validation failed for revisions: " + ", ".join(invalid_revs), file=sys.stderr)
        sys.exit(4)

    # ------------------------------------------------------------------
    # Determine CLI suffix (once) for locating result directories
    # ------------------------------------------------------------------
    cli_suffix_part = ""
    if os.getenv("BRKBCH_CLI"):
        cli_suffix_part = "-" + pathlib.Path(os.getenv("BRKBCH_CLI")).name

    # ------------------------------------------------------------------
    # Analyse existing results to compute average llmMillis per revision
    # ------------------------------------------------------------------
    avg_llm_millis: Dict[str, float] = {}
    totals: Dict[str, List[float]] = defaultdict(list)

    # Traverse existing results directories (up to the requested --runs count)
    for run_number in range(1, args.runs + 1):
        results_dir = (
            pathlib.Path(f"{args.mode}results{cli_suffix_part}")
            / f"{project_path.name}{run_number}"
        )
        if not results_dir.is_dir():
            continue

        for json_file in results_dir.glob("*.json"):
            # Parse the JSON; ignore unreadable/broken files
            try:
                with open(json_file, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception:
                continue

            # Extract revision from filename (model-revision.json)
            rev_from_name = json_file.stem.rsplit("-", 1)[-1]
            if rev_from_name not in revisions:
                continue

            llm_val = data.get("llmMillis")
            if isinstance(llm_val, (int, float)):
                totals[rev_from_name].append(float(llm_val))

    # Compute simple averages
    for rev in revisions:
        values = totals.get(rev)
        if values:
            avg_llm_millis[rev] = sum(values) / len(values)
        else:
            avg_llm_millis[rev] = 0.0

    # Sort revisions by average time (descending).  For equal averages, retain
    # the original ordering from stdin via Python's stable sort.
    revisions.sort(key=lambda r: avg_llm_millis.get(r, 0.0), reverse=True)

    # ------------------------------------------------------------------
    # Create list of (revision, model) pairs to run
    # ------------------------------------------------------------------
    models = [m.strip() for m in args.model.split(",") if m.strip()]

    jobs_to_run: List[tuple[str, str, int]] = []
    processed_jobs_count = 0
    for run_number in range(1, args.runs + 1):
        results_dir_name = f"{project_path.name}{run_number}"
        results_dir = (
            pathlib.Path(f"{args.mode}results{cli_suffix_part}") / results_dir_name
        )

        for rev in revisions:
            for model in models:
                results_path = results_dir / f"{model}-{rev}.json"
                if results_path.exists():
                    processed_jobs_count += 1
                else:
                    jobs_to_run.append((rev, model, run_number))

    # ------------------------------------------------------------------
    # Report outstanding (unattempted) tasks per model, by run
    # ------------------------------------------------------------------
    model_task_counts_by_run: dict[int, dict[str, int]] = {}
    for _, model, run_number in jobs_to_run:
        if run_number not in model_task_counts_by_run:
            model_task_counts_by_run[run_number] = {m: 0 for m in models}
        model_task_counts_by_run[run_number][model] += 1

    for run_number in range(1, args.runs + 1):
        run_counts = model_task_counts_by_run.get(run_number, {})
        total_tasks_for_run = sum(run_counts.values())

        print(f"--- Run {run_number} ---", file=sys.stderr)
        print(f"{total_tasks_for_run} Tasks Remaining", file=sys.stderr)
        for model in sorted(models):  # alphabetize models
            count = run_counts.get(model, 0)
            print(f"  {model}: {count}", file=sys.stderr)
        print("", file=sys.stderr) # Add a newline between runs

    if args.dry_run:
        sys.exit(0)

    if not jobs_to_run:
        print(
            "All provided revision/model combinations have already been processed.",
            file=sys.stderr,
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Run all jobs in parallel
    # ------------------------------------------------------------------
    results: List[RunResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_job = {
            executor.submit(run_with_retries, args.project, rev, model, args.mode, cli_suffix_part, run_number, jvm_args, args.stagger, args.exclude): (
                rev,
                model,
                run_number,
            )
            for rev, model, run_number in jobs_to_run
        }
        try:
            for future in concurrent.futures.as_completed(future_to_job):
                rev, model, run_number = future_to_job[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    # Fatal error – re-raise so the program exits with traceback
                    print(
                        f"Fatal error while processing revision '{rev}' with model '{model}' (run {run_number}): {exc}",
                        file=sys.stderr,
                    )
                    raise
        finally:
            executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Aggregate exit status
    # ------------------------------------------------------------------
    if any(r == RunResult.AGENT_FAILED for r in results):
        print("Agent failed for some revisions.", file=sys.stderr)
        sys.exit(2)
    elif any(r == RunResult.TESTS_FAILED for r in results):
        print("Tests failed for some revisions.", file=sys.stderr)
        sys.exit(1)
    else:
        print("All revisions processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
