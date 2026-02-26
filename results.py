#!/usr/bin/env python3
"""
Graph benchmark results.

Given a directory that contains JSON files named {model}-{revision}.json,
this script aggregates multi-run task results per (project, revision, model)
under rerun-failed semantics and computes the score

    score = 1 / log2(sum_build_failures + 2)

for tasks whose final observed run is successful (else score 0). It validates
task hashes across models and drops incomplete hashes before scoring.

Usage:
    uv run results.py /path/to/benchmark/dir [--no-estimate] [--filter taskcommits-0126/openround.jsonl]
"""

import argparse
import json
import math
import os
import sys
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import litellm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tasks
from irt_token_impute import IRTTokenImputer
from litellm import completion_cost
from litellm.utils import CostPerToken
litellm.suppress_debug_info = True

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
    message=r".*default value for l1_ratios.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
    message=r"'.*' was deprecated in version 1\.8.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
    message=r"The fitted attributes of LogisticRegressionCV will be simplified.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Treat the new Tool classes introduced in v1\.5 as experimental for now; the API and rcParam may change in future versions\.",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\._multiprocessing_helpers",
    message=r"\[Errno 13\] Permission denied\.\s+joblib will operate in serial mode",
)

# Load pretty names for models from model_metadata.json in the results directory (if available).
# This avoids hardcoding display names and keeps chart labels consistent.
_MODEL_METADATA_PATH: Path | None = None
_MODEL_ALIASES: dict[str, str] = {}
_PROJECT_METADATA_PATH: Path | None = None
_PROJECT_LANGUAGES: dict[str, str] = {}


def _configure_model_metadata_path(results_dir: Path) -> None:
    """
    Point model metadata loading at the selected results directory.
    Falls back to empty aliases if metadata is missing.
    """
    global _MODEL_METADATA_PATH, _MODEL_ALIASES
    _MODEL_METADATA_PATH = results_dir / "model_metadata.json"
    model_metadata = json.loads(_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    _MODEL_ALIASES = model_metadata.get("model_aliases", {}) or {}


def _configure_project_metadata_path(results_dir: Path) -> None:
    """
    Load project metadata from project_metadata.json in the selected results directory.
    """
    global _PROJECT_METADATA_PATH, _PROJECT_LANGUAGES
    _PROJECT_METADATA_PATH = results_dir / "project_metadata.json"
    if not _PROJECT_METADATA_PATH.is_file():
        _PROJECT_LANGUAGES = {}
        print(
            f"ERROR: project language metadata not found at {_PROJECT_METADATA_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)
        return

    try:
        project_metadata = json.loads(_PROJECT_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        _PROJECT_LANGUAGES = {}
        print(
            f"ERROR: failed to parse {_PROJECT_METADATA_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_mapping = project_metadata
    if isinstance(project_metadata, dict):
        raw_mapping = (
            project_metadata.get("projects", {})
            or project_metadata.get("language_by_project", {})
            or project_metadata
        )
    if not isinstance(raw_mapping, dict):
        raw_mapping = {}
        print(
            f"ERROR: invalid schema in {_PROJECT_METADATA_PATH}; expected a project->language mapping",
            file=sys.stderr,
        )
        sys.exit(1)

    _PROJECT_LANGUAGES = {
        str(k): str(v)
        for k, v in raw_mapping.items()
        if isinstance(k, str) and isinstance(v, str)
    }


def _project_language(project_name: str) -> str:
    """
    Resolve a project language label from metadata.
    """
    return _PROJECT_LANGUAGES.get(project_name, "unknown")

def _pretty_name(model_alias: str) -> str:
    """
    Return a human-friendly name for a model alias using model_metadata.json.
    Falls back to the alias if no mapping is present.
    """
    try:
        return _MODEL_ALIASES.get(model_alias, model_alias)
    except Exception:
        return model_alias

# Alias -> litellm model name
MODEL_MAPPING = {
    "o3": "o3",
    "o3-high": "o3",

    "gp2.5-default": "gemini/gemini-2.5-pro",
    "gp2.5-high": "gemini/gemini-2.5-pro",
    "gem3pp": "gemini/gemini-3-pro-preview",

    "v3": "deepseek/deepseek-v3",
    "ds-v3.1": "deepseek/deepseek-v3",

    "o4-mini": "o4-mini",
    "o4-mini-high": "o4-mini",

    "sonnet4-nothink": "claude-4-sonnet-20250514",
    "sonnet4": "claude-4-sonnet-20250514",
    "sonnet4-high": "claude-4-sonnet-20250514",

    "flash-2.0": "gemini/gemini-2.0-flash",

    "flash2.5-0925": "gemini/gemini-2.5-flash",
    "flash-2.5": "gemini/gemini-2.5-flash",
    "flash-2.5-high": "gemini/gemini-2.5-flash",
    "flash-2.5-nothink": "gemini/gemini-2.5-flash",

    "flash3-instant": "flash3",
    "flash3-low": "flash3",
    "gem3fnt": "flash3",
    "gem3fhi": "flash3",

    'gpt5-high': 'gpt5',
    'gpt5-nothink': 'gpt5',

    'gpt5-mini-nothink': 'gpt5-mini',
    'gpt5-mini-low': 'gpt5-mini',
    'gpt5-mini-high': 'gpt5-mini',
    'gpt5.1-codex-mini': 'gpt5-mini',

    'gpt5.1-high': 'gpt5.1',
    'gpt5.1-low': 'gpt5.1',
    'gpt5.1-nothink': 'gpt5.1',

    'gpt5.2': 'GPT-5.2',
    'gpt5.2-high': 'GPT-5.2',
    'gpt5.2-instant': 'GPT-5.2',

    'gpt5-nano-high': 'gpt5-nano',

    "opus4.1-high": 'opus4.1',
    "opus4.1-nothink": 'opus4.1',

    "dsv3.2-exp": "dsv3.2",
    "dsr3.2-exp": "dsr3.2",
    "devstral2": "mistral/devstral-2512",

    "haiku4.5-instant": "haiku4.5",

    "opus4.6": 'opus4.5',
    "opus4.6-instant": 'opus4.5',
    "sonnet4.6": 'sonnet4.5',
    "sonnet4.6-instant": 'sonnet4.5',

    "g3.1p": 'g3p',
}

# Custom per-token pricing for models not recognised by LiteLLM
CUSTOM_MODEL_PRICING: dict[str, CostPerToken] = {
    "sonnet4.5":    {"input_cost_per_token": 3e-06,  "output_cost_per_token": 1.5e-05},
    "sonnet4.5-high": {"input_cost_per_token": 3e-06,  "output_cost_per_token": 1.5e-05},
    "haiku4.5":    {"input_cost_per_token": 1e-06,  "output_cost_per_token": 5e-06},
    "dsv3.2":       {"input_cost_per_token": 2.8e-07,  "output_cost_per_token": 4.2e-07},
    "dsr3.2":       {"input_cost_per_token": 2.8e-07,  "output_cost_per_token": 4.2e-07},
    "m2":           {"input_cost_per_token": 3e-07,  "output_cost_per_token": 1.2e-06},
    "m2.5":         {"input_cost_per_token": 3e-07, "output_cost_per_token": 1.2e-06},
    "k2":           {"input_cost_per_token": 1e-06,  "output_cost_per_token": 3e-06},
    "k2.5":         {"input_cost_per_token": 6e-07, "output_cost_per_token": 3e-06},
    "k2-thinking":  {"input_cost_per_token": 1e-06,  "output_cost_per_token": 2.5e-06},
    "gpt-oss-20b":  {"input_cost_per_token": 5e-08,"output_cost_per_token": 2e-07},
    "gpt-oss-120b": {"input_cost_per_token": 1e-07,"output_cost_per_token": 7.5e-07},
    "grok-3":       {"input_cost_per_token": 3e-06,  "output_cost_per_token": 1.5e-6},
    "grok-3-mini":  {"input_cost_per_token": 3e-07,  "output_cost_per_token": 5e-7},
    "grok-3-mini-high":  {"input_cost_per_token": 3e-07,  "output_cost_per_token": 5e-7},
    "gcf1":         {"input_cost_per_token": 2e-07,"output_cost_per_token": 1.5e-06},
    "grok4-fast":   {"input_cost_per_token": 2e-07,"output_cost_per_token": 5e-07},
    "grok4.1-fast": {"input_cost_per_token": 2e-07,"output_cost_per_token": 5e-07},
    "grok4":        {"input_cost_per_token": 3e-06,"output_cost_per_token": 1.5e-05},
    "opus4.1":      {"input_cost_per_token": 1.5e-05,"output_cost_per_token": 7.5e-5},
    "opus4.5":      {"input_cost_per_token": 5e-06,"output_cost_per_token": 2.5e-5},
    "sonnet4.6-instant": {"input_cost_per_token": 3e-06,  "output_cost_per_token": 15e-06},
    "q3-27b":      {"input_cost_per_token": 0.3e-06,  "output_cost_per_token": 2.4e-06},
    "q3-35b":       {"input_cost_per_token": 0.25e-06,  "output_cost_per_token": 2e-06},
    "q3-397b":       {"input_cost_per_token": 0.6e-06,  "output_cost_per_token": 3.6e-06},
    "q3c":          {"input_cost_per_token": 1.5e-06,  "output_cost_per_token": 7.5e-06},
    "q3c-next":     {"input_cost_per_token": 1.2e-07, "output_cost_per_token": 7.5e-07},
    "q3c-fp8":      {"input_cost_per_token": 4e-07,  "output_cost_per_token": 1.6e-06},
    "q3c-30b":      {"input_cost_per_token": 1e-07,  "output_cost_per_token": 3e-07},
    "q3next":       {"input_cost_per_token": 1.5e-07,  "output_cost_per_token": 1.5e-06},
    "qwen3-max":    {"input_cost_per_token": 1.2e-06,  "output_cost_per_token": 6e-06},
    "mistral/devstral-2512": {"input_cost_per_token": 4e-07,"output_cost_per_token": 2e-06},
    "devstral2-small": {"input_cost_per_token": 1e-07, "output_cost_per_token": 3e-07},
    "glm5":         {"input_cost_per_token": 1e-06,   "output_cost_per_token": 3.2e-06},
    "glm4.7-flash": {"input_cost_per_token": 0.06e-06, "output_cost_per_token": 0.4e-06},
    "step3.5-flash": {"input_cost_per_token": 0.1e-06, "output_cost_per_token": 0.3e-06},
    "glm4.5":       {"input_cost_per_token": 6e-07,  "output_cost_per_token": 2.2e-06},
    "glm4.6":       {"input_cost_per_token": 6e-07,  "output_cost_per_token": 2.2e-06},
    "glm4.6-fp8":   {"input_cost_per_token": 4.5e-07,  "output_cost_per_token": 2.0e-06},
    "glm4.5-air":   {"input_cost_per_token": 2e-07,  "output_cost_per_token": 1.1e-06},
    "r1":           {"input_cost_per_token": 5.5e-07,  "output_cost_per_token": 2.19e-06},
    "nemo3":        {"input_cost_per_token": 5e-08,  "output_cost_per_token": 2e-07},
    "ds-r1.1":      {"input_cost_per_token": 5.5e-07,  "output_cost_per_token": 2.19e-06},
    "gpt5":         {"input_cost_per_token": 1.25e-06,"output_cost_per_token": 1e-05},
    "gpt5.1":       {"input_cost_per_token": 1.25e-06,"output_cost_per_token": 1e-05},
    "GPT-5.2":      {"input_cost_per_token": 1.25e-06,"output_cost_per_token": 1e-05},
    "gpt5-codex":   {"input_cost_per_token": 1.25e-06,"output_cost_per_token": 1e-05},
    "gpt5-mini":    {"input_cost_per_token": 2.5e-07,"output_cost_per_token": 2e-06},
    "gpt5-nano":    {"input_cost_per_token": 5e-08,"output_cost_per_token": 4e-07},
    "gpt5.2":       {"input_cost_per_token": 1.75e-06,"output_cost_per_token": 1.4e-05},
    "g3p":          {"input_cost_per_token": 2e-06,  "output_cost_per_token": 1.2e-05},
    "flash3":       {"input_cost_per_token": 5e-07,  "output_cost_per_token": 3e-06},
}

MODEL_PRICING_OVER_200k: dict[str, CostPerToken] = {
    "g3p": {"input_cost_per_token": 4.0e-06,  "output_cost_per_token": 1.8e-05},
    "opus4.5": {"input_cost_per_token": 10e-06, "output_cost_per_token": 37.5e-06},
    "sonnet4.5": {"input_cost_per_token": 6e-06, "output_cost_per_token": 22.5e-06},
}

CALIBRATION_VERSION = "irt-median-ratio-v1"
CALIBRATION_MIN_POINTS_PER_MODEL = 8
CALIBRATION_FACTOR_MIN = 0.5
CALIBRATION_FACTOR_MAX = 2.0
FILTER_ESTIMATOR_STRATA = 5
FILTER_ESTIMATOR_SHRINKAGE_STRENGTH = 8.0

# Palette used for model bars. Cycles these colors as necessary.
_MODEL_BAR_PALETTE = [
    "#EB3F33",  # Primary red
    "#3B82F6",  # Blue
    "#10B981",  # Green
    "#F59E0B",  # Amber
    "#8B5CF6",  # Purple
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#6B7280",  # Gray
    "#F87171",  # Light red
    "#60A5FA",  # Light blue
    "#34D399",  # Light green
    "#FBBF24",  # Light amber
]

_ANSI_RESET = "\x1b[0m"


def _ansi_color_supported() -> bool:
    """Return True when terminal ANSI colors are likely desirable."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if not sys.stdout.isatty():
        return False
    term = (os.environ.get("TERM") or "").lower()
    return term != "" and term != "dumb"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` hex color into an (r, g, b) tuple."""
    clean = color.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(ch * 2 for ch in clean)
    if len(clean) != 6:
        raise ValueError(f"Unsupported color format: {color}")
    return (
        int(clean[0:2], 16),
        int(clean[2:4], 16),
        int(clean[4:6], 16),
    )


def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """Convert RGB to approximate xterm-256 color index."""
    return (
        16
        + (int(round(r / 255 * 5)) * 36)
        + (int(round(g / 255 * 5)) * 6)
        + int(round(b / 255 * 5))
    )


def _colorize(text: str, color: str | None, *, enabled: bool = False) -> str:
    """Wrap ``text`` in ANSI color escape using the provided hex color."""
    if not enabled or not color:
        return text
    try:
        r, g, b = _hex_to_rgb(color)
        return f"\x1b[38;5;{_rgb_to_ansi256(r, g, b)}m{text}{_ANSI_RESET}"
    except Exception:
        return text


def _get_distinct_colors(n: int) -> list[str]:
    """
    Return `n` colours from the model-bar palette, cycling as required.

    The palette is intentionally explicit and will be repeated to supply as
    many distinct colours as requested.
    """
    repeats = (n + len(_MODEL_BAR_PALETTE) - 1) // len(_MODEL_BAR_PALETTE)
    return (_MODEL_BAR_PALETTE * repeats)[:n]


def _read_filter_rows(lines: Iterable[str]) -> list[tuple[str, float | None]]:
    """
    Parse filter rows as `(revision_hash, difficulty)` tuples.

    Supports legacy plain-token lines and JSONL rows with a ``hash`` field.
    Difficulty may be omitted.
    """
    rows: list[tuple[str, float | None]] = []
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                token = stripped.split(maxsplit=1)[0]
                if not token:
                    raise ValueError(
                        f"Invalid revision spec in filter file line {line_num}: {stripped}"
                    ) from exc
                rows.append((token.lower(), None))
                continue

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid filter row at line {line_num}: expected JSON object"
                )

            revision_hash = payload.get("hash")
            if not isinstance(revision_hash, str) or not revision_hash.strip():
                raise ValueError(
                    f"Missing or invalid 'hash' field in filter file line {line_num}"
                )
            raw_difficulty = payload.get("difficulty")
            if raw_difficulty is None:
                difficulty = None
            else:
                try:
                    parsed = float(raw_difficulty)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid 'difficulty' in filter file line {line_num}: {raw_difficulty}"
                    ) from exc
                if not math.isfinite(parsed):
                    raise ValueError(
                        f"Invalid 'difficulty' in filter file line {line_num}: {raw_difficulty}"
                    )
                difficulty = min(1.0, max(0.0, parsed))
            rows.append((revision_hash.strip().lower(), difficulty))
    return rows


def _load_filter_revisions_and_difficulty(filter_file: Path | None) -> tuple[set[str], dict[str, float]]:
    """
    Load filter hashes and optional difficulty metadata from file.
    """
    if filter_file is None:
        return set(), {}
    with filter_file.open("r", encoding="utf-8") as fp:
        rows = _read_filter_rows(fp)
        revisions = {revision for revision, _ in rows}
        difficulties = {
            revision: difficulty for revision, difficulty in rows if difficulty is not None
        }
        return revisions, difficulties


def _load_filter_revisions(filter_file: Path | None) -> set[str] | None:
    """
    Backward-compatible filter-revision loader.
    """
    revisions, _difficulties = _load_filter_revisions_and_difficulty(filter_file)
    return revisions if revisions else None


def _on_key_press(event):
    """
    Handle key press events to allow quitting with Ctrl+Q / Cmd+Q (macOS) and
    their equivalents across platforms.

    Matplotlib encodes modifier keys directly into the ``event.key`` string in the
    form ``"<modifier>+<key>"`` – e.g. ``"ctrl+q"`` or ``"cmd+q"``.  We therefore
    simply normalise this string to lower-case and look for any combination that
    represents the familiar quit shortcut.
    """
    key_combo = (event.key or "").lower()
    if key_combo in {
        "ctrl+q",
        "cmd+q",
        "super+q",
        "meta+q",
        "ctrl+shift+q",
        "cmd+shift+q",
        "super+shift+q",
        "meta+shift+q",
    }:
        plt.close("all")


def _discover_projects(base_directory: Path) -> list[str]:
    """
    Infer project prefixes from run directory names of the form ``<project><run_number>``.

    Returns a sorted list of unique project names discovered under ``base_directory``.
    """
    return tasks.discover_projects(base_directory)


def _parse_args() -> tuple[
    list[tuple[str, Path]],
    list[str] | None,
    set[str] | None,
    dict[str, float] | None,
    set[str] | None,
    list[str],
    int | None,
    bool,
    set[str],
    bool,
    str,
    Path | None,
    bool,
]:
    """
    Parse command-line arguments.

    Returns
    -------
    tuple
        (run_info, models_order, exclude_models, filter_task_difficulty, filter_revisions, projects, tasksize, text, charts, nolegend, json_export_dir, no_estimate) where:
          - run_info is a list of (project, Path) tuples, one per run directory
          - models_order is a list of model names supplied via ``--models`` (in order, with "_" placeholders) or ``None``
          - exclude_models is a set of model names supplied via ``--exclude`` / ``--no-models`` or ``None``
          - filter_task_difficulty maps revision hashes supplied via ``--filter`` to optional difficulty values in ``[0,1]``; missing values default to 0.5.
            ``None`` means no ``--filter`` was used.
          - filter_revisions is a set of revision hashes supplied via ``--filter`` or ``None``
          - projects is the list of project name prefixes
          - tasksize is an integer maximum token count (or None if not provided)
          - text is a boolean; when True, prints ASCII charts to stdout instead of using matplotlib
        - charts is a set of chart names to render (e.g. {'mainscore', 'score_v_latency'})
        - nolegend is a boolean; when True, legends are suppressed on all charts
        - no_estimate disables solve-all metric estimation and uses raw total costs/latencies.
        - exclude_missing controls how missing run coverage is handled before scoring: ``tasks`` (default) or ``models``
        - json_export_dir is the optional JSON output directory path when export is enabled.
    """
    parser = argparse.ArgumentParser(description="Graph benchmark results.")
    parser.add_argument(
        "base_directory",
        type=Path,
        help="Base directory containing project run directories (e.g. coderesults/).",
    )
    parser.add_argument(
        "--projects",
        metavar="LIST",
        help="Comma-separated list of project names, prefixes for run directories (e.g. brokk,gizmo). "
        "If omitted, projects are auto-discovered from numeric-suffixed run directory names.",
    )
    parser.add_argument(
        "--runs",
        metavar="LIST",
        help="Comma-separated list of run numbers to include (e.g. 1,2,4). "
        "If omitted, all numeric-suffixed runs for the selected projects are processed.",
    )
    parser.add_argument(
        "--models",
        metavar="LIST",
        help="Comma-separated list of models to include (e.g. o3,flash-2.5). "
        "If omitted, all models in the directory are processed.",
    )
    parser.add_argument(
        "--exclude",
        metavar="LIST",
        help="Comma-separated list of models to exclude (e.g. gpt-4-turbo). "
        "Exclusion takes precedence over inclusion.",
    )
    parser.add_argument(
        "--no-models",
        metavar="LIST",
        help="Comma-separated list of models to exclude from the full model set (all models except these).",
    )
    parser.add_argument(
        "--filter",
        metavar="FILE",
        help=(
            "Path to a file with one revision hash per line or JSONL rows like "
            "{'hash': '...', 'difficulty': 0.0..1.0}. Only listed hashes are considered."
        ),
    )
    parser.add_argument(
        "--tasksize",
        metavar="N",
        type=int,
        help="Maximum task size in tokens; if provided only revisions with tokens <= N are considered.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Output ASCII bar charts to stdout instead of rendering with matplotlib.",
    )
    parser.add_argument(
        "--charts",
        metavar="LIST",
        default="mainscore,score_v_latency,score_v_spend,score_speed_price",
        help="Comma-separated list of charts to render. "
        "Available: mainscore, per_task, llm_runtime, score_v_spend, score_v_latency, score_speed_price, by_task_length. "
        "Default: mainscore,score_v_latency,score_v_spend,score_speed_price",
    )
    parser.add_argument(
        "--nolegend",
        action="store_true",
        help="Suppress legend display on all charts.",
    )
    parser.add_argument(
        "--exclude-missing",
        metavar="MODE",
        choices={"tasks", "models"},
        default="tasks",
        help=(
            "When set to 'tasks' (default), drop tasks that are missing model coverage. "
            "When set to 'models', drop models that are missing results for any selected task."
        ),
    )
    parser.add_argument(
        "--json",
        metavar="DIR",
        help=(
            "Enable JSONL export and write under the target results/webapp-json directory. "
            "Files are written to parent/webapp-json/<DIR> for the selected results directory."
        ),
    )
    parser.add_argument(
        "--no-estimate",
        action="store_true",
        help="Use raw total cost/latency values (including failed tasks) instead of estimated solve-all metrics.",
    )
    args = parser.parse_args()

    if not args.base_directory.is_dir():
        parser.error(f"{args.base_directory} is not a directory")

    if args.projects:
        projects = [p.strip() for p in str(args.projects).split(",") if p.strip()]
        if not projects:
            parser.error("--projects must be a comma-separated list of one or more project names.")
    else:
        projects = _discover_projects(args.base_directory)
        if not projects:
            parser.error(
                "No project run directories found in "
                f"{args.base_directory}; pass --projects explicitly."
            )

    charts = {c.strip() for c in str(args.charts).split(",") if c.strip()}
    if not charts:
        parser.error("--charts provided but no valid chart names parsed")
    valid_charts = {
        "mainscore",
        "per_task",
        "llm_runtime",
        "score_v_spend",
        "score_v_latency",
        "score_speed_price",
        "by_task_length",
    }
    invalid_charts = charts - valid_charts
    if invalid_charts:
        parser.error(f"Invalid chart names: {', '.join(sorted(invalid_charts))}. Valid options: {', '.join(sorted(valid_charts))}")

    run_info: list[tuple[str, Path]] = []
    if args.runs:
        try:
            run_numbers = sorted(
                [int(r.strip()) for r in args.runs.split(",") if r.strip()]
            )
        except ValueError:
            parser.error("--runs must be a comma-separated list of integers.")

        if not run_numbers:
            parser.error("--runs provided but no valid run numbers parsed")

        for project_name in projects:
            for run_num in run_numbers:
                run_dir = args.base_directory / f"{project_name}{run_num}"
                if not run_dir.is_dir():
                    parser.error(f"Run directory {run_dir} does not exist.")
                run_info.append((project_name, run_dir))
    else:
        # Autodiscover runs per project
        for project_name in projects:
            for d in args.base_directory.iterdir():
                if not d.is_dir():
                    continue
                discovered_project, run_suffix = tasks.parse_project_and_run(d)
                if discovered_project == project_name and run_suffix is not None:
                    run_info.append((project_name, d))

        if not run_info:
            parser.error(
                f"No run directories found for projects {', '.join(projects)} "
                f"in {args.base_directory}"
            )
        # Sort by project then run dir name
        run_info.sort(key=lambda pr: (pr[0], pr[1].name))

    models_order: list[str] | None = None
    if args.models:
        models_order = [m.strip() for m in args.models.split(",")]
        if not models_order:
            parser.error("--models provided but no valid model names parsed")

    exclude_models: set[str] | None = None
    if args.exclude:
        exclude_models = {m.strip() for m in args.exclude.split(",") if m.strip()}
        if not exclude_models:
            parser.error("--exclude provided but no valid model names parsed")

    if args.no_models:
        no_models = {m.strip() for m in args.no_models.split(",") if m.strip()}
        if not no_models:
            parser.error("--no-models provided but no valid model names parsed")
        if exclude_models is None:
            exclude_models = no_models
        else:
            exclude_models = exclude_models.union(no_models)

    filter_revisions: set[str] | None = None
    filter_task_difficulty: dict[str, float] | None = None
    if args.filter:
        filter_path = Path(args.filter).expanduser()
        if not filter_path.is_file():
            parser.error(f"--filter file not found: {filter_path}")
        try:
            filter_revisions, filter_task_difficulty = _load_filter_revisions_and_difficulty(filter_path)
        except (OSError, ValueError) as exc:
            parser.error(f"--filter file could not be parsed: {exc}")

    tasksize: int | None = None
    if args.tasksize is not None:
        if args.tasksize < 0:
            parser.error("--tasksize must be non-negative")
        tasksize = int(args.tasksize)

    json_export_dir: Path | None = None
    if args.json is not None:
        json_export_dir = Path(args.json).expanduser()

    return (
        run_info,
        models_order,
        exclude_models,
        filter_task_difficulty,
        filter_revisions,
        projects,
        tasksize,
        bool(args.text),
        charts,
        bool(args.nolegend),
        args.exclude_missing,
        json_export_dir,
        bool(args.no_estimate),
    )


def _compute_cost_from_tokens(
    model: str,
    revision: str,
    input_tokens: float,
    output_tokens: float,
    cached_input_tokens: float = 0.0,
) -> float:
    """
    Compute model spend from token counts using the same pricing path as result files.

    Token values are coerced to non-negative integers because pricing is token-based.
    """
    prompt_tokens = tasks.safe_nonnegative_int(round(tasks.safe_nonnegative_float(input_tokens)))
    completion_tokens = tasks.safe_nonnegative_int(round(tasks.safe_nonnegative_float(output_tokens)))
    cache_read_tokens = tasks.safe_nonnegative_int(round(tasks.safe_nonnegative_float(cached_input_tokens)))

    response_for_cost = {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_read_input_tokens": cache_read_tokens,
        }
    }

    mapped_model = MODEL_MAPPING.get(model, model)
    pricing_to_use = None

    if prompt_tokens > 200000 and mapped_model in MODEL_PRICING_OVER_200k:
        pricing_to_use = MODEL_PRICING_OVER_200k[mapped_model]
    elif mapped_model in CUSTOM_MODEL_PRICING:
        pricing_to_use = CUSTOM_MODEL_PRICING[mapped_model]

    try:
        if pricing_to_use is not None:
            cost = completion_cost(
                completion_response=response_for_cost,
                model=mapped_model,
                custom_cost_per_token=pricing_to_use,
                custom_pricing=True,
            )
        else:
            cost = completion_cost(
                completion_response=response_for_cost,
                model=mapped_model,
            )
    except Exception as exc:
        if "Provider NOT provided" in str(exc) or "LLM Provider NOT provided" in str(exc):
            print(
                f"Error: Pricing not found for model '{model}'. Add model pricing to CUSTOM_MODEL_PRICING or MODEL_MAPPING.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if "Pricing not found" in str(exc):
            print(
                f"Error: Pricing not found for model '{model}'. Add model pricing to CUSTOM_MODEL_PRICING or MODEL_MAPPING.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        raise

    return float(cost or 0.0)


def _compute_cost_for_result(model: str, revision: str, data: dict) -> float:
    """Compute cost for a single result payload."""
    return _compute_cost_from_tokens(
        model=model,
        revision=revision,
        input_tokens=tasks.safe_nonnegative_int(data.get("inputTokens", 0)),
        output_tokens=tasks.safe_nonnegative_int(data.get("outputTokens", 0)),
        cached_input_tokens=tasks.safe_nonnegative_int(data.get("cachedInputTokens", 0)),
    )


def _load_tokens_by_revision(
    projects: list[str],
    run_results: list[tuple[str, int, Path, dict[str, dict[str, dict]]]],
    revisions: Iterable[str],
) -> dict[str, int]:
    """Load codetask prompt-token counts for each composite revision."""
    project_codetasks: dict[str, Path] = {}
    for project_name in projects:
        candidate_runs = [run_dir for (p, _run_num, run_dir, _res) in run_results if p == project_name]
        if not candidate_runs:
            continue
        base_dir = candidate_runs[0].parent
        candidates = [base_dir / "codetasks", base_dir.parent / "codetasks"]
        codetasks_dir = next((c for c in candidates if c.is_dir()), None)
        if codetasks_dir is not None:
            project_codetasks[project_name] = codetasks_dir

    tokens_by_rev: dict[str, int] = {}
    for comp_rev in revisions:
        if ":" in comp_rev:
            proj, rev = comp_rev.split(":", 1)
        else:
            proj, rev = (projects[0] if projects else ""), comp_rev
        codetasks_dir = project_codetasks.get(proj)
        if codetasks_dir is None:
            continue
        task_file = codetasks_dir / f"{rev}.json"
        if not task_file.is_file():
            continue
        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
            tokens_val = data.get("tokens")
            if isinstance(tokens_val, int) and tokens_val >= 0:
                tokens_by_rev[comp_rev] = tokens_val
        except Exception:
            continue

    return tokens_by_rev


def _clip_calibration_factor(raw_factor: float, *, model: str, factor_name: str) -> float:
    """Validate and clip a calibration factor to configured safety bounds."""
    if not math.isfinite(raw_factor) or raw_factor <= 0:
        raise ValueError(
            f"Invalid {factor_name} calibration factor for model '{model}': {raw_factor}"
        )
    return float(np.clip(raw_factor, CALIBRATION_FACTOR_MIN, CALIBRATION_FACTOR_MAX))


def _fit_token_calibration_factors(
    imputed_rows: Iterable[dict[str, object]],
    models: list[str],
    *,
    min_points_per_model: int = CALIBRATION_MIN_POINTS_PER_MODEL,
    strict: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Fit per-model multiplicative calibration for input/output token predictions.

    Uses only successful rows with positive observed and predicted token counts.
    """
    input_ratios: dict[str, list[float]] = {m: [] for m in models}
    output_ratios: dict[str, list[float]] = {m: [] for m in models}

    for row in imputed_rows:
        model = str(row["model"])
        if model not in input_ratios:
            continue
        if not bool(int(row.get("success", 0))):
            continue

        observed_in = tasks.safe_nonnegative_float(row.get("input_tokens", 0.0))
        observed_out = tasks.safe_nonnegative_float(row.get("output_tokens", 0.0))
        expected_in = tasks.safe_nonnegative_float(row.get("expected_input_tokens_if_solved", 0.0))
        expected_out = tasks.safe_nonnegative_float(row.get("expected_output_tokens_if_solved", 0.0))

        if observed_in > 0 and expected_in > 0:
            input_ratios[model].append(observed_in / expected_in)
        if observed_out > 0 and expected_out > 0:
            output_ratios[model].append(observed_out / expected_out)

    input_factors: dict[str, float] = {}
    output_factors: dict[str, float] = {}
    for model in models:
        if len(input_ratios[model]) < min_points_per_model:
            if not strict:
                print(
                    f"Warning: skipping token calibration for model '{model}' due to insufficient input points "
                    f"(need {min_points_per_model}, got {len(input_ratios[model])})",
                    file=sys.stderr,
                )
                continue
            raise ValueError(
                f"Cannot calibrate input tokens for model '{model}': "
                f"need at least {min_points_per_model} points, got {len(input_ratios[model])}"
            )
        if len(output_ratios[model]) < min_points_per_model:
            if not strict:
                print(
                    f"Warning: skipping token calibration for model '{model}' due to insufficient output points "
                    f"(need {min_points_per_model}, got {len(output_ratios[model])})",
                    file=sys.stderr,
                )
                continue
            raise ValueError(
                f"Cannot calibrate output tokens for model '{model}': "
                f"need at least {min_points_per_model} points, got {len(output_ratios[model])}"
            )

        input_factors[model] = _clip_calibration_factor(
            float(np.median(input_ratios[model])),
            model=model,
            factor_name="input-token",
        )
        output_factors[model] = _clip_calibration_factor(
            float(np.median(output_ratios[model])),
            model=model,
            factor_name="output-token",
        )

    return input_factors, output_factors


def _apply_token_calibration(
    imputed_rows: Iterable[dict[str, object]],
    input_factors: dict[str, float],
    output_factors: dict[str, float],
) -> list[dict[str, object]]:
    """Apply per-model token calibration factors to imputed expected token columns."""
    calibrated_rows: list[dict[str, object]] = []
    for row in imputed_rows:
        model = str(row["model"])
        calibrated = dict(row)
        expected_in = tasks.safe_nonnegative_float(row.get("expected_input_tokens_if_solved", 0.0))
        expected_out = tasks.safe_nonnegative_float(row.get("expected_output_tokens_if_solved", 0.0))
        calibrated["calibrated_expected_input_tokens_if_solved"] = expected_in * input_factors[model]
        calibrated["calibrated_expected_output_tokens_if_solved"] = expected_out * output_factors[model]
        calibrated_rows.append(calibrated)
    return calibrated_rows


def _fit_latency_calibration_factors(
    calibrated_rows: Iterable[dict[str, object]],
    models: list[str],
    latency_ms_per_token: dict[str, float],
    *,
    min_points_per_model: int = CALIBRATION_MIN_POINTS_PER_MODEL,
    strict: bool = True,
) -> dict[str, float]:
    """
    Fit per-model multiplicative calibration for predicted LLM latency.

    Calibrates predicted success-row latency against observed success-row latency.
    """
    ratio_by_model: dict[str, list[float]] = {m: [] for m in models}
    for row in calibrated_rows:
        model = str(row["model"])
        if model not in ratio_by_model:
            continue
        if not bool(int(row.get("success", 0))):
            continue

        predicted_in = tasks.safe_nonnegative_float(
            row.get("calibrated_expected_input_tokens_if_solved", 0.0)
        )
        predicted_out = tasks.safe_nonnegative_float(
            row.get("calibrated_expected_output_tokens_if_solved", 0.0)
        )
        predicted_llm_ms = (predicted_in + predicted_out) * tasks.safe_nonnegative_float(
            latency_ms_per_token[model]
        )
        observed_llm_ms = tasks.safe_nonnegative_float(row.get("llm_ms", 0.0))
        if predicted_llm_ms > 0 and observed_llm_ms > 0:
            ratio_by_model[model].append(observed_llm_ms / predicted_llm_ms)

    factors: dict[str, float] = {}
    for model in models:
        if len(ratio_by_model[model]) < min_points_per_model:
            if not strict:
                print(
                    f"Warning: skipping latency calibration for model '{model}' due to insufficient points "
                    f"(need {min_points_per_model}, got {len(ratio_by_model[model])})",
                    file=sys.stderr,
                )
                continue
            raise ValueError(
                f"Cannot calibrate latency for model '{model}': "
                f"need at least {min_points_per_model} points, got {len(ratio_by_model[model])}"
            )
        factors[model] = _clip_calibration_factor(
            float(np.median(ratio_by_model[model])),
            model=model,
            factor_name="latency",
        )
    return factors


def _ratio_map(values: dict[str, float], baseline: str, models: list[str]) -> dict[str, float]:
    """Compute value ratios relative to the given baseline model."""
    baseline_value = tasks.safe_nonnegative_float(values.get(baseline, 0.0))
    if baseline_value <= 0:
        raise ValueError(f"Baseline '{baseline}' has non-positive value ({baseline_value})")
    return {m: tasks.safe_nonnegative_float(values.get(m, 0.0)) / baseline_value for m in models}


def _compute_slice_ratio_diagnostics(
    rev_scores: dict[str, dict[str, dict]],
    revisions: Iterable[str],
    models: list[str],
    estimated_cost_by_task: dict[tuple[str, str], float],
) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
    """
    Compute synthetic-vs-observed spend ratio diagnostics for canonical slices.

    Returns a tuple:
      - diagnostics by slice name
      - per-model mean absolute percentage error across all applicable slices
    """
    diagnostics: dict[str, dict[str, object]] = {}
    per_model_errors: dict[str, list[float]] = {}
    available_models = set(models)
    revision_list = list(revisions)

    slice_specs = [
        ("gcf1_success", "gcf1", ["gpt5.2-nothink", "flash3-nothink", "haiku4.5", "gcf1"], "gcf1"),
        ("haiku_success", "haiku4.5", ["haiku4.5", "flash3-nothink", "gpt5.2-nothink"], "haiku4.5"),
    ]

    for slice_name, selector_model, slice_models, baseline_model in slice_specs:
        if any(m not in available_models for m in slice_models):
            missing = [m for m in slice_models if m not in available_models]
            diagnostics[slice_name] = {
                "status": "skipped",
                "reason": f"missing models: {', '.join(missing)}",
                "n_tasks": 0,
            }
            continue

        selected_revisions = [
            rev for rev in revision_list if bool(rev_scores.get(rev, {}).get(selector_model, {}).get("final_success", False))
        ]
        if not selected_revisions:
            diagnostics[slice_name] = {
                "status": "skipped",
                "reason": f"no tasks where '{selector_model}' succeeded",
                "n_tasks": 0,
            }
            continue

        observed_totals = {m: 0.0 for m in slice_models}
        estimated_totals = {m: 0.0 for m in slice_models}
        for comp_rev in selected_revisions:
            for model in slice_models:
                observed_totals[model] += tasks.safe_nonnegative_float(
                    rev_scores.get(comp_rev, {}).get(model, {}).get("cost", 0.0)
                )
                task_key = (comp_rev, model)
                if task_key not in estimated_cost_by_task:
                    raise ValueError(
                        "Missing estimated counterfactual cost for "
                        f"{comp_rev}/{model} in slice '{slice_name}'"
                    )
                estimated_totals[model] += tasks.safe_nonnegative_float(estimated_cost_by_task[task_key])

        observed_ratios = _ratio_map(observed_totals, baseline_model, slice_models)
        estimated_ratios = _ratio_map(estimated_totals, baseline_model, slice_models)

        errors_by_model: dict[str, float] = {}
        errors: list[float] = []
        for model in slice_models:
            if model == baseline_model:
                continue
            observed_ratio = tasks.safe_nonnegative_float(observed_ratios.get(model, 0.0))
            if observed_ratio <= 0:
                raise ValueError(
                    f"Observed ratio is non-positive for model '{model}' in slice '{slice_name}'."
                )
            estimated_ratio = tasks.safe_nonnegative_float(estimated_ratios.get(model, 0.0))
            error = abs(estimated_ratio - observed_ratio) / observed_ratio
            errors_by_model[model] = error
            errors.append(error)
            per_model_errors.setdefault(model, []).append(error)

        diagnostics[slice_name] = {
            "status": "ok",
            "selector_model": selector_model,
            "baseline_model": baseline_model,
            "models": slice_models,
            "n_tasks": len(selected_revisions),
            "observed_ratios": observed_ratios,
            "estimated_ratios": estimated_ratios,
            "abs_pct_error_by_model": errors_by_model,
            "mean_abs_pct_error": float(np.mean(errors)) if errors else 0.0,
        }

    per_model_mean_error = {
        model: (float(np.mean(errs)) if errs else float("nan"))
        for model, errs in per_model_errors.items()
    }
    return diagnostics, per_model_mean_error


def _print_calibration_diagnostics(
    models: list[str],
    input_factors: dict[str, float],
    output_factors: dict[str, float],
    latency_factors: dict[str, float],
    per_model_slice_error: dict[str, float],
    slice_diagnostics: dict[str, dict[str, object]],
) -> None:
    """Print calibration factors and slice-level diagnostics in text mode."""
    print("# Calibration Diagnostics")
    print(f"Version: {CALIBRATION_VERSION}")
    max_label = max((len(_pretty_name(m)) for m in models), default=0)
    for model in models:
        model_error = per_model_slice_error.get(model)
        error_text = f"{model_error * 100.0:.2f}%" if isinstance(model_error, float) and math.isfinite(model_error) else "n/a"
        print(
            f"{_pretty_name(model):<{max_label}} | "
            f"in={input_factors.get(model, float('nan')):.3f} "
            f"out={output_factors.get(model, float('nan')):.3f} "
            f"lat={latency_factors.get(model, float('nan')):.3f} "
            f"slice_err={error_text}"
        )

    for slice_name in sorted(slice_diagnostics.keys()):
        diag = slice_diagnostics[slice_name]
        status = str(diag.get("status", "unknown"))
        if status != "ok":
            print(f"{slice_name}: skipped ({diag.get('reason', 'unknown reason')})")
            continue
        mean_err = tasks.safe_nonnegative_float(diag.get("mean_abs_pct_error", 0.0))
        n_tasks = int(diag.get("n_tasks", 0))
        print(f"{slice_name}: n={n_tasks}, mean_abs_pct_error={mean_err * 100.0:.2f}%")
    print("")


def _latency_ms_per_token_from_successes(
    rev_scores: dict[str, dict[str, dict]],
    models: list[str],
    revisions: Iterable[str],
    *,
    strict: bool = True,
) -> dict[str, float]:
    """
    Estimate per-model LLM milliseconds-per-token from successful aggregated tasks.

    Raises a hard error if any model lacks enough successful-token/runtime data.
    """
    ratios: dict[str, float] = {}
    for model in models:
        total_llm_ms = 0.0
        total_tokens = 0.0
        for comp_rev in revisions:
            metrics = rev_scores.get(comp_rev, {}).get(model)
            if not metrics or not bool(metrics.get("final_success", False)):
                continue
            total_llm_ms += tasks.safe_nonnegative_float(metrics.get("llm", 0.0))
            total_tokens += tasks.safe_nonnegative_float(metrics.get("input_tokens", 0.0))
            total_tokens += tasks.safe_nonnegative_float(metrics.get("output_tokens", 0.0))

        if total_llm_ms <= 0.0 or total_tokens <= 0.0:
            if not strict:
                print(
                    f"Warning: skipping latency-rate calibration for model '{model}' due to no successful runtime/token data.",
                    file=sys.stderr,
                )
                continue
            raise ValueError(
                "Cannot estimate solve-all latency: model "
                f"'{model}' lacks successful runtime/token data."
            )

        ratios[model] = total_llm_ms / total_tokens

    return ratios


def _counterfactual_metrics_from_imputed_rows(
    imputed_rows: Iterable[dict[str, object]],
    rev_scores: dict[str, dict[str, dict]],
    models: list[str],
    latency_ms_per_token: dict[str, float],
    latency_factors: dict[str, float],
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, float], dict[tuple[str, str], float]]:
    """
    Compute counterfactual per-task/model costs and LLM latency.

    Successful tasks use observed tokens, while failed tasks use imputed
    ``*_if_solved`` token counts.
    """
    estimated_total_spend = {m: 0.0 for m in models}
    estimated_cost_by_task: dict[tuple[str, str], float] = {}
    estimated_total_latency_seconds = {m: 0.0 for m in models}
    estimated_latency_by_task_seconds: dict[tuple[str, str], float] = {}

    for row in imputed_rows:
        model = str(row["model"])
        task_id = str(row["task_id"])
        success = bool(int(row["success"]))

        if success:
            input_tokens = tasks.safe_nonnegative_float(row.get("input_tokens", 0.0))
            output_tokens = tasks.safe_nonnegative_float(row.get("output_tokens", 0.0))
            cached_input_tokens = tasks.safe_nonnegative_float(
                rev_scores.get(task_id, {}).get(model, {}).get("cached_input_tokens", 0.0)
            )
        else:
            input_tokens = tasks.safe_nonnegative_float(
                row.get("calibrated_expected_input_tokens_if_solved", row.get("expected_input_tokens_if_solved", 0.0))
            )
            output_tokens = tasks.safe_nonnegative_float(
                row.get("calibrated_expected_output_tokens_if_solved", row.get("expected_output_tokens_if_solved", 0.0))
            )
            cached_input_tokens = 0.0

        cost = _compute_cost_from_tokens(
            model=model,
            revision=task_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )
        if success:
            llm_ms = tasks.safe_nonnegative_float(rev_scores.get(task_id, {}).get(model, {}).get("llm", 0.0))
        else:
            llm_ms = (
                (input_tokens + output_tokens)
                * tasks.safe_nonnegative_float(latency_ms_per_token.get(model, 0.0))
                * tasks.safe_nonnegative_float(latency_factors.get(model, 1.0))
            )

        estimated_total_spend[model] += cost
        estimated_cost_by_task[(task_id, model)] = cost
        llm_s = llm_ms / 1000.0
        estimated_total_latency_seconds[model] += llm_s
        estimated_latency_by_task_seconds[(task_id, model)] = llm_s

    return (
        estimated_total_spend,
        estimated_cost_by_task,
        estimated_total_latency_seconds,
        estimated_latency_by_task_seconds,
    )


def _task_hash_from_revision(revision: str) -> str:
    """Extract the hash component from a composite revision key."""
    if ":" in revision:
        return revision.split(":", 1)[1]
    return revision


def _coerce_difficulty_by_revision(
    revisions: Iterable[str],
    difficulty_by_revision: dict[str, float] | None,
) -> dict[str, float]:
    """Map model-agnostic revision IDs to difficulty scores, defaulting missing values."""
    difficulties: dict[str, float] = {}
    source: dict[str, float] = difficulty_by_revision or {}
    for revision in revisions:
        by_hash = source.get(_task_hash_from_revision(revision))
        by_full = source.get(revision)
        difficulty = by_hash if by_hash is not None else by_full
        if not isinstance(difficulty, (float, int)):
            difficulty = 0.5
        difficulties[revision] = float(min(1.0, max(0.0, difficulty)))
    return difficulties


def _build_difficulty_strata(
    task_difficulty: dict[str, float],
    n_strata: int = FILTER_ESTIMATOR_STRATA,
) -> dict[str, int]:
    """Assign each task to a difficulty stratum in [0, n_strata)."""
    task_ids = list(task_difficulty.keys())
    if not task_ids or n_strata <= 1:
        return {task_id: 0 for task_id in task_ids}

    values = np.array([task_difficulty[task_id] for task_id in task_ids], dtype=float)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_strata + 1))
    # Guard against duplicate edges with little samples.
    edges = np.clip(edges, 0.0, 1.0)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6

    strata: dict[str, int] = {}
    for task_id in task_ids:
        diff = task_difficulty.get(task_id, 0.5)
        idx = int(np.searchsorted(edges, diff, side="right") - 1)
        idx = max(0, min(n_strata - 1, idx))
        strata[task_id] = idx
    return strata


def _shrink_to_prior(
    value: float,
    sample_count: int,
    prior: float,
    *,
    strength: float = FILTER_ESTIMATOR_SHRINKAGE_STRENGTH,
) -> float:
    """Blend an estimate toward a prior with strength-controlled shrinkage."""
    if not math.isfinite(value):
        return prior
    if not math.isfinite(prior):
        return value
    if sample_count <= 0 or strength <= 0:
        return value
    weight = sample_count / (sample_count + strength)
    return (weight * value) + ((1.0 - weight) * prior)


def _mean_or_nan(total: float, count: int) -> float:
    return (total / count) if count > 0 else float("nan")


def _compute_estimated_solve_all_metrics_for_filter_set(
    rev_scores: dict[str, dict[str, dict]],
    models: list[str],
    revisions: Iterable[str],
    tokens_by_rev: dict[str, int],
    difficulty_by_revision: dict[str, float] | None,
) -> tuple[
    dict[str, float],
    dict[tuple[str, str], float],
    dict[str, float],
    dict[tuple[str, str], float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, dict[str, object]],
    dict[str, float],
]:
    """
    Estimate solve-all metrics for a filter set using stratified shrinkage.

    This path avoids IRT and focuses on bias-aware estimates when task count is
    intentionally small/stratified.
    """
    revision_list = list(revisions)

    task_difficulty = _coerce_difficulty_by_revision(revision_list, difficulty_by_revision)
    task_strata = _build_difficulty_strata(task_difficulty, n_strata=FILTER_ESTIMATOR_STRATA)
    if not task_strata:
        print(
            "Warning: skipping estimated solve-all metrics for filtered set: "
            "cannot derive task difficulty strata.",
            file=sys.stderr,
        )
        return ({}, {}, {}, {}, {}, {}, {}, {}, {})

    rows: list[dict[str, object]] = []
    for comp_rev in revision_list:
        prompt_tokens = int(tokens_by_rev[comp_rev])
        model_data = rev_scores.get(comp_rev, {})
        for model in models:
            metrics = model_data.get(model)
            if not metrics:
                raise ValueError(
                    f"Cannot run sparse filtered metric estimation: missing metrics for model '{model}' revision '{comp_rev}'."
                )

            final_success = bool(metrics.get("final_success", False))
            rows.append(
                {
                    "model": model,
                    "task_id": comp_rev,
                    "prompt_tokens": prompt_tokens,
                    "difficulty": task_difficulty.get(comp_rev, 0.5),
                    "difficulty_stratum": int(task_strata.get(comp_rev, 0)),
                    "success": 1 if final_success else 0,
                    "input_tokens": tasks.safe_nonnegative_float(metrics.get("input_tokens", 0.0)) if final_success else float("nan"),
                    "output_tokens": tasks.safe_nonnegative_float(metrics.get("output_tokens", 0.0)) if final_success else float("nan"),
                    "llm_ms": tasks.safe_nonnegative_float(metrics.get("llm", 0.0)) if final_success else float("nan"),
                }
            )

    if not rows:
        print(
            "Warning: skipping estimated solve-all metrics for filtered set: "
            "no rows available.",
            file=sys.stderr,
        )
        return ({}, {}, {}, {}, {}, {}, {}, {}, {})

    # Aggregate success evidence by model and difficulty stratum.
    model_input_sum_by_stratum: dict[str, dict[int, float]] = {}
    model_input_count_by_stratum: dict[str, dict[int, int]] = {}
    model_output_sum_by_stratum: dict[str, dict[int, float]] = {}
    model_output_count_by_stratum: dict[str, dict[int, int]] = {}
    model_rate_sum_by_stratum: dict[str, dict[int, float]] = {}
    model_rate_count_by_stratum: dict[str, dict[int, int]] = {}

    model_input_sum: dict[str, float] = {}
    model_input_count: dict[str, int] = {}
    model_output_sum: dict[str, float] = {}
    model_output_count: dict[str, int] = {}
    model_rate_sum: dict[str, float] = {}
    model_rate_count: dict[str, int] = {}

    global_input_sum_by_stratum: dict[int, float] = {}
    global_input_count_by_stratum: dict[int, int] = {}
    global_output_sum_by_stratum: dict[int, float] = {}
    global_output_count_by_stratum: dict[int, int] = {}
    global_rate_sum_by_stratum: dict[int, float] = {}
    global_rate_count_by_stratum: dict[int, int] = {}

    global_input_sum = 0.0
    global_input_count = 0
    global_output_sum = 0.0
    global_output_count = 0
    global_rate_sum = 0.0
    global_rate_count = 0

    for row in rows:
        if not bool(int(row["success"])):
            continue
        model = str(row["model"])
        prompt_tokens = max(1.0, tasks.safe_nonnegative_float(row.get("prompt_tokens", 1.0)))
        stratum = int(row.get("difficulty_stratum", 0))

        observed_in = tasks.safe_nonnegative_float(row.get("input_tokens", 0.0))
        observed_out = tasks.safe_nonnegative_float(row.get("output_tokens", 0.0))
        observed_llm = tasks.safe_nonnegative_float(row.get("llm_ms", 0.0))

        input_ratio = observed_in / prompt_tokens
        output_ratio = observed_out / prompt_tokens
        total_tokens = observed_in + observed_out

        model_input_sum_by_stratum.setdefault(model, {})
        model_input_count_by_stratum.setdefault(model, {})
        model_input_sum_by_stratum[model][stratum] = model_input_sum_by_stratum[model].get(stratum, 0.0) + input_ratio
        model_input_count_by_stratum[model][stratum] = model_input_count_by_stratum[model].get(stratum, 0) + 1
        model_output_sum_by_stratum.setdefault(model, {})
        model_output_count_by_stratum.setdefault(model, {})
        model_output_sum_by_stratum[model][stratum] = model_output_sum_by_stratum[model].get(stratum, 0.0) + output_ratio
        model_output_count_by_stratum[model][stratum] = model_output_count_by_stratum[model].get(stratum, 0) + 1

        model_input_sum[model] = model_input_sum.get(model, 0.0) + input_ratio
        model_input_count[model] = model_input_count.get(model, 0) + 1
        model_output_sum[model] = model_output_sum.get(model, 0.0) + output_ratio
        model_output_count[model] = model_output_count.get(model, 0) + 1

        if total_tokens > 0:
            llm_rate = observed_llm / total_tokens
            model_rate_sum_by_stratum.setdefault(model, {})
            model_rate_count_by_stratum.setdefault(model, {})
            model_rate_sum_by_stratum[model][stratum] = model_rate_sum_by_stratum[model].get(stratum, 0.0) + llm_rate
            model_rate_count_by_stratum[model][stratum] = model_rate_count_by_stratum[model].get(stratum, 0) + 1
            model_rate_sum[model] = model_rate_sum.get(model, 0.0) + llm_rate
            model_rate_count[model] = model_rate_count.get(model, 0) + 1

            global_rate_sum_by_stratum[stratum] = global_rate_sum_by_stratum.get(stratum, 0.0) + llm_rate
            global_rate_count_by_stratum[stratum] = global_rate_count_by_stratum.get(stratum, 0) + 1
            global_rate_sum += llm_rate
            global_rate_count += 1

        global_input_sum_by_stratum[stratum] = global_input_sum_by_stratum.get(stratum, 0.0) + input_ratio
        global_input_count_by_stratum[stratum] = global_input_count_by_stratum.get(stratum, 0) + 1
        global_output_sum_by_stratum[stratum] = global_output_sum_by_stratum.get(stratum, 0.0) + output_ratio
        global_output_count_by_stratum[stratum] = global_output_count_by_stratum.get(stratum, 0) + 1
        global_input_sum += input_ratio
        global_input_count += 1
        global_output_sum += output_ratio
        global_output_count += 1

    if global_input_count == 0 or global_output_count == 0:
        print(
            "Warning: skipping estimated solve-all metrics for filtered set: "
            "no successful tasks available for token or token-ratio estimation.",
            file=sys.stderr,
        )
        return ({}, {}, {}, {}, {}, {}, {}, {}, {})

    global_input_overall = global_input_sum / global_input_count
    global_output_overall = global_output_sum / global_output_count
    global_rate_overall = _mean_or_nan(global_rate_sum, global_rate_count)

    global_input_by_stratum = {
        s: _mean_or_nan(total, global_input_count_by_stratum.get(s, 0))
        for s, total in global_input_sum_by_stratum.items()
    }
    global_output_by_stratum = {
        s: _mean_or_nan(total, global_output_count_by_stratum.get(s, 0))
        for s, total in global_output_sum_by_stratum.items()
    }
    global_rate_by_stratum = {
        s: _mean_or_nan(total, global_rate_count_by_stratum.get(s, 0))
        for s, total in global_rate_sum_by_stratum.items()
    }

    def _mean_by_stratum(
        sum_by_stratum: dict[str, dict[int, float]],
        count_by_stratum: dict[str, dict[int, int]],
        model: str,
        stratum: int,
    ) -> tuple[float, int]:
        return (
            _mean_or_nan(
                sum_by_stratum.get(model, {}).get(stratum, 0.0),
                count_by_stratum.get(model, {}).get(stratum, 0),
            ),
            count_by_stratum.get(model, {}).get(stratum, 0),
        )

    def _mean_overall(
        sum_by_model: dict[str, float],
        count_by_model: dict[str, int],
        model: str,
    ) -> tuple[float, int]:
        return (
            _mean_or_nan(sum_by_model.get(model, 0.0), count_by_model.get(model, 0)),
            count_by_model.get(model, 0),
        )

    imputed_rows: list[dict[str, object]] = []
    for row in rows:
        model = str(row["model"])
        success = bool(int(row["success"]))
        if success:
            imputed_rows.append(row)
            continue

        stratum = int(row.get("difficulty_stratum", 0))
        prompt_tokens = max(1.0, tasks.safe_nonnegative_float(row.get("prompt_tokens", 1.0)))

        model_input_stratum, model_input_n = _mean_by_stratum(
            model_input_sum_by_stratum,
            model_input_count_by_stratum,
            model,
            stratum,
        )
        model_output_stratum, model_output_n = _mean_by_stratum(
            model_output_sum_by_stratum,
            model_output_count_by_stratum,
            model,
            stratum,
        )
        model_rate_stratum, model_rate_n = _mean_by_stratum(
            model_rate_sum_by_stratum,
            model_rate_count_by_stratum,
            model,
            stratum,
        )

        model_input_overall, model_input_total_n = _mean_overall(model_input_sum, model_input_count, model)
        model_output_overall, model_output_total_n = _mean_overall(model_output_sum, model_output_count, model)
        model_rate_overall, model_rate_total_n = _mean_overall(model_rate_sum, model_rate_count, model)

        pred_in_ratio = _shrink_to_prior(
            _shrink_to_prior(
                model_input_stratum,
                model_input_n,
                global_input_by_stratum.get(stratum, global_input_overall),
                strength=FILTER_ESTIMATOR_SHRINKAGE_STRENGTH,
            ),
            model_input_total_n,
            global_input_overall,
            strength=FILTER_ESTIMATOR_SHRINKAGE_STRENGTH,
        )
        pred_out_ratio = _shrink_to_prior(
            _shrink_to_prior(
                model_output_stratum,
                model_output_n,
                global_output_by_stratum.get(stratum, global_output_overall),
                strength=FILTER_ESTIMATOR_SHRINKAGE_STRENGTH,
            ),
            model_output_total_n,
            global_output_overall,
            strength=FILTER_ESTIMATOR_SHRINKAGE_STRENGTH,
        )
        pred_rate = _shrink_to_prior(
            _shrink_to_prior(
                model_rate_stratum,
                model_rate_n,
                global_rate_by_stratum.get(stratum, global_rate_overall),
                strength=FILTER_ESTIMATOR_SHRINKAGE_STRENGTH,
            ),
            model_rate_total_n,
            global_rate_overall if math.isfinite(global_rate_overall) else float("nan"),
            strength=FILTER_ESTIMATOR_SHRINKAGE_STRENGTH,
        )

        if not math.isfinite(pred_rate):
            pred_rate = _shrink_to_prior(
                global_rate_overall,
                0,
                float("nan"),
                strength=1.0,
            )
        if not math.isfinite(pred_rate):
            pred_rate = 0.0

        pred_input = max(0.0, pred_in_ratio * prompt_tokens)
        pred_output = max(0.0, pred_out_ratio * prompt_tokens)

        row = dict(row)
        row["expected_input_tokens_if_solved"] = pred_input
        row["expected_output_tokens_if_solved"] = pred_output
        row["llm_ms"] = (pred_input + pred_output) * pred_rate
        imputed_rows.append(row)

    latency_ms_per_token: dict[str, float] = {}
    for model in models:
        _, model_rate_total_n = _mean_overall(model_rate_sum, model_rate_count, model)
        model_rate = _mean_overall(model_rate_sum, model_rate_count, model)[0]
        if not math.isfinite(model_rate):
            model_rate = _mean_or_nan(global_rate_sum, global_rate_count)
        latency_ms_per_token[model] = _shrink_to_prior(
            model_rate,
            model_rate_total_n,
            _mean_or_nan(global_rate_sum, global_rate_count),
            strength=FILTER_ESTIMATOR_SHRINKAGE_STRENGTH,
        )

    (
        estimated_total_spend,
        estimated_cost_by_task,
        estimated_total_latency_seconds,
        estimated_latency_by_task_seconds,
    ) = _counterfactual_metrics_from_imputed_rows(
        imputed_rows=imputed_rows,
        rev_scores=rev_scores,
        models=models,
        latency_ms_per_token=latency_ms_per_token,
        latency_factors={},
    )

    return (
        estimated_total_spend,
        estimated_cost_by_task,
        estimated_total_latency_seconds,
        estimated_latency_by_task_seconds,
        {},
        {},
        {},
        {},
        {},
    )


def _compute_estimated_solve_all_metrics(
    rev_scores: dict[str, dict[str, dict]],
    models: list[str],
    revisions: Iterable[str],
    tokens_by_rev: dict[str, int],
    filter_task_difficulty: dict[str, float] | None = None,
) -> tuple[
    dict[str, float],
    dict[tuple[str, str], float],
    dict[str, float],
    dict[tuple[str, str], float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, dict[str, object]],
    dict[str, float],
]:
    """
    Estimate solve-all spend and latency.

    In filter mode (when difficulty metadata is supplied), uses a
    stratified two-phase shrinkage estimator instead of IRT.
    Raises a hard error only when required task-token metadata is missing or the
    imputation input is structurally incomplete.
    """
    revision_list = list(revisions)
    missing_tokens = [rev for rev in revision_list if rev not in tokens_by_rev]
    if missing_tokens:
        sample = ", ".join(missing_tokens[:5])
        suffix = "..." if len(missing_tokens) > 5 else ""
        raise ValueError(
            "Cannot run IRT metric imputation: missing codetasks token data for "
            f"{len(missing_tokens)} revision(s): {sample}{suffix}"
        )

    if filter_task_difficulty is not None:
        print(
            "Using filtered-run stratified shrinkage estimator (IRT disabled) for solve-all metrics.",
            file=sys.stderr,
        )
        return _compute_estimated_solve_all_metrics_for_filter_set(
            rev_scores=rev_scores,
            models=models,
            revisions=revision_list,
            tokens_by_rev=tokens_by_rev,
            difficulty_by_revision=filter_task_difficulty,
        )

    rows: list[dict[str, object]] = []
    for comp_rev in revision_list:
        prompt_tokens = int(tokens_by_rev[comp_rev])
        model_data = rev_scores.get(comp_rev, {})
        for model in models:
            metrics = model_data.get(model)
            if not metrics:
                raise ValueError(
                    f"Cannot run IRT metric imputation: missing metrics for model '{model}' revision '{comp_rev}'."
                )

            final_success = bool(metrics.get("final_success", False))
            rows.append(
                {
                    "model": model,
                    "task_id": comp_rev,
                    "prompt_tokens": prompt_tokens,
                    "success": 1 if final_success else 0,
                    "input_tokens": tasks.safe_nonnegative_float(metrics.get("input_tokens", 0.0)) if final_success else float("nan"),
                    "output_tokens": tasks.safe_nonnegative_float(metrics.get("output_tokens", 0.0)) if final_success else float("nan"),
                    "llm_ms": tasks.safe_nonnegative_float(metrics.get("llm", 0.0)) if final_success else float("nan"),
                }
            )

    fit_df = pd.DataFrame(rows)
    try:
        imputer = IRTTokenImputer(random_state=0).fit(
            fit_df,
            model_col="model",
            task_col="task_id",
            prompt_tokens_col="prompt_tokens",
            success_col="success",
            input_tokens_col="input_tokens",
            output_tokens_col="output_tokens",
        )
    except ValueError as exc:
        print(
            "Warning: skipping estimated solve-all metrics for filtered set: "
            f"insufficient diversity for IRT fitting ({exc})",
            file=sys.stderr,
        )
        return ({}, {}, {}, {}, {}, {}, {}, {}, {})
    imputed = imputer.impute_tokens_if_solved(fit_df)
    imputed = imputed.merge(
        fit_df[["model", "task_id", "success", "input_tokens", "output_tokens", "llm_ms"]],
        on=["model", "task_id"],
        how="left",
    )
    imputed_rows = imputed.to_dict("records")
    input_factors, output_factors = _fit_token_calibration_factors(
        imputed_rows,
        models,
        strict=False,
    )
    calibratable_token_models = sorted(
        set(input_factors.keys()) & set(output_factors.keys())
    )
    if not calibratable_token_models:
        print(
            "Warning: no models have enough token calibration data in this filtered set; "
            "estimated solve-all metrics will be reported as unavailable (nan).",
            file=sys.stderr,
        )
        return ({}, {}, {}, {}, {}, {}, {}, {}, {})

    skipped_token_models = sorted(set(models) - set(calibratable_token_models))
    if skipped_token_models:
        print(
            "Warning: skipping estimated solve-all metrics for models without enough token calibration data: "
            + ", ".join(skipped_token_models),
            file=sys.stderr,
        )

    token_calibrated_rows = [row for row in imputed_rows if str(row["model"]) in set(calibratable_token_models)]
    calibrated_rows = _apply_token_calibration(token_calibrated_rows, input_factors, output_factors)
    latency_ms_per_token = _latency_ms_per_token_from_successes(
        rev_scores,
        calibratable_token_models,
        revision_list,
        strict=False,
    )
    latency_factors = _fit_latency_calibration_factors(
        calibrated_rows=calibrated_rows,
        models=calibratable_token_models,
        latency_ms_per_token=latency_ms_per_token,
        strict=False,
    )
    calibratable_latency_models = sorted(
        set(calibratable_token_models) & set(latency_factors.keys())
    )
    if not calibratable_latency_models:
        print(
            "Warning: no models have enough latency calibration data in this filtered set; "
            "latency outputs will be reported as unavailable (nan).",
            file=sys.stderr,
        )
        calibratable_latency_models = calibratable_token_models

    skipped_latency_models = sorted(set(calibratable_token_models) - set(calibratable_latency_models))
    if skipped_latency_models:
        print(
            "Warning: skipping estimated solve-all metrics for models without enough latency calibration data: "
            + ", ".join(skipped_latency_models),
            file=sys.stderr,
        )

    valid_calibration_models = set(calibratable_latency_models)
    calibrated_rows = [
        row for row in calibrated_rows if str(row["model"]) in valid_calibration_models
    ]

    (
        estimated_total_spend,
        estimated_cost_by_task,
        estimated_total_latency_seconds,
        estimated_latency_by_task_seconds,
    ) = _counterfactual_metrics_from_imputed_rows(
        imputed_rows=calibrated_rows,
        rev_scores=rev_scores,
        models=calibratable_latency_models,
        latency_ms_per_token=latency_ms_per_token,
        latency_factors=latency_factors,
    )
    slice_diagnostics, per_model_slice_error = _compute_slice_ratio_diagnostics(
        rev_scores=rev_scores,
        revisions=revision_list,
        models=calibratable_latency_models,
        estimated_cost_by_task=estimated_cost_by_task,
    )

    return (
        estimated_total_spend,
        estimated_cost_by_task,
        estimated_total_latency_seconds,
        estimated_latency_by_task_seconds,
        {m: input_factors[m] for m in calibratable_latency_models if m in input_factors},
        {m: output_factors[m] for m in calibratable_latency_models if m in output_factors},
        {m: latency_factors[m] for m in calibratable_latency_models if m in latency_factors},
        slice_diagnostics,
        per_model_slice_error,
    )


def _compute_estimated_solve_all_spend(
    rev_scores: dict[str, dict[str, dict]],
    models: list[str],
    revisions: Iterable[str],
    tokens_by_rev: dict[str, int],
    filter_task_difficulty: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """
    Backward-compatible spend-only wrapper over ``_compute_estimated_solve_all_metrics``.
    """
    (
        estimated_spend,
        estimated_cost_by_task,
        _estimated_latency,
        _estimated_latency_by_task,
        _input_factors,
        _output_factors,
        _latency_factors,
        _slice_diagnostics,
        _per_model_slice_error,
    ) = _compute_estimated_solve_all_metrics(
        rev_scores=rev_scores,
        models=models,
        revisions=revisions,
        tokens_by_rev=tokens_by_rev,
        filter_task_difficulty=filter_task_difficulty,
    )
    return estimated_spend, estimated_cost_by_task


def _aggregate_run(
    directory: Path,
    run_number: int,
    include_models: set[str] | None = None,
    exclude_models: set[str] | None = None,
    tasksize: int | None = None,
    filter_revisions: set[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Read all JSON files inside one run directory.

    Returns ``dict[revision][model] = run_metrics`` where metrics represent one
    observed execution for that specific run number.
    """
    revisions_by_model: dict[str, set[str]] = {}
    rev_scores: dict[str, dict[str, dict]] = {}

    codetasks_dir = next(
        (
            c
            for c in (directory.parent / "codetasks", directory.parent.parent / "codetasks")
            if c.is_dir()
        ),
        None,
    )

    for json_file in directory.glob("*.json"):
        if "tasktune" in json_file.name.lower():
            continue
        model = tasks.model_from_result_filename(json_file)
        if include_models is not None and model not in include_models:
            continue
        if exclude_models is not None and model in exclude_models:
            continue

        _parsed_model, revision = tasks.parse_result_filename(json_file.name)
        if revision is None:
            stem = json_file.stem
            _m, _sep, revision_guess = stem.rpartition("-")
            revision = revision_guess if revision_guess else stem
        revision = revision.lower()
        if filter_revisions is not None and revision not in filter_revisions:
            continue

        file_data = tasks.read_json_object(json_file)
        if file_data is None:
            print(f"Skipping {json_file}: invalid JSON", file=sys.stderr)
            continue

        if tasksize is not None:
            if codetasks_dir is None:
                print(
                    f"Warning: Could not locate codetasks/ for run directory {directory}; skipping revision '{revision}' due to --tasksize.",
                    file=sys.stderr,
                )
                continue
            task_file = codetasks_dir / f"{revision}.json"
            if not task_file.is_file():
                print(
                    f"Warning: Missing codetasks file {task_file}; skipping revision '{revision}' due to --tasksize.",
                    file=sys.stderr,
                )
                continue
            try:
                task_data = json.loads(task_file.read_text(encoding="utf-8"))
                tokens_val = task_data.get("tokens")
                if not isinstance(tokens_val, int) or tokens_val < 0:
                    print(
                        f"Warning: Invalid tokens value in {task_file}; skipping revision '{revision}' due to --tasksize.",
                        file=sys.stderr,
                    )
                    continue
                if tokens_val > tasksize:
                    continue
            except Exception:
                print(
                    f"Warning: Failed to read tokens from {task_file}; skipping revision '{revision}' due to --tasksize.",
                    file=sys.stderr,
                )
                continue

        revisions_by_model.setdefault(model, set()).add(revision)

        run_metrics = tasks.extract_run_metrics(model, revision, file_data, run_number)
        run_metrics["cost"] = _compute_cost_for_result(model, revision, tasks.extract_result_payload(file_data))

        rev_scores.setdefault(revision, {})[model] = run_metrics

    if not revisions_by_model:
        raise ValueError(
            "No benchmark files matched the given criteria "
            f"in {directory}"
        )

    return rev_scores


def _validate_task_hashes(
    task_runs: dict[str, dict[str, dict[int, dict]]],
    models: list[str],
    max_run_number: int,
) -> tuple[list[str], dict[str, set[str]]]:
    """
    Validate per-task model coverage and post-failure continuity.

    Returns ``(valid_revisions, discarded_reasons)`` where ``discarded_reasons``
    maps composite revision -> set of reason strings.
    """
    discarded: dict[str, set[str]] = {}
    valid: list[str] = []

    for comp_rev in sorted(task_runs.keys()):
        reasons: set[str] = set()
        model_runs = task_runs[comp_rev]

        for model in models:
            runs_for_model = model_runs.get(model)
            if not runs_for_model:
                reasons.add(f"missing_model_data:{model}")
                continue

            for run_n in sorted(runs_for_model.keys()):
                if run_n >= max_run_number:
                    continue
                run_metrics = runs_for_model[run_n]
                previous_outcome = (
                    tasks.RUN_OUTCOME_SUCCESS
                    if bool(run_metrics.get("success", False))
                    else tasks.RUN_OUTCOME_AGENT_FAILED
                )
                if (
                    tasks.should_run_in_rerun_mode("failed", run_n + 1, previous_outcome)
                    and (run_n + 1) not in runs_for_model
                ):
                    reasons.add(f"missing_post_failure_run:{model}:run{run_n}")
                    break

        if reasons:
            discarded[comp_rev] = reasons
        else:
            valid.append(comp_rev)

    return valid, discarded


def _validate_models(
    task_runs: dict[str, dict[str, dict[int, dict]]],
    models: list[str],
    max_run_number: int,
) -> tuple[list[str], dict[str, set[str]]]:
    """
    Validate per-model task coverage and post-failure continuity.

    Returns ``(valid_models, discarded_reasons)`` where ``discarded_reasons``
    maps model name -> set of reason strings.
    """
    discarded: dict[str, set[str]] = {}
    valid: list[str] = []

    revision_keys = sorted(task_runs.keys())
    for model in models:
        reasons: set[str] = set()
        for comp_rev in revision_keys:
            runs_for_model = task_runs.get(comp_rev, {}).get(model)
            if not runs_for_model:
                reasons.add(f"missing_task_data:{comp_rev}")
                break

            for run_n in sorted(runs_for_model.keys()):
                if run_n >= max_run_number:
                    continue
                run_metrics = runs_for_model[run_n]
                previous_outcome = (
                    tasks.RUN_OUTCOME_SUCCESS
                    if bool(run_metrics.get("success", False))
                    else tasks.RUN_OUTCOME_AGENT_FAILED
                )
                if (
                    tasks.should_run_in_rerun_mode("failed", run_n + 1, previous_outcome)
                    and (run_n + 1) not in runs_for_model
                ):
                    reasons.add(f"missing_post_failure_run:{model}:run{run_n}")
                    break

            if reasons:
                break

        if reasons:
            discarded[model] = reasons
        else:
            valid.append(model)

    return valid, discarded


def _aggregate_task_runs(
    task_runs: dict[str, dict[str, dict[int, dict]]],
    revisions: Iterable[str],
    models: list[str],
) -> dict[str, dict[str, dict]]:
    """
    Aggregate observed per-run metrics into one task score per model.

    Metrics are always summed across observed runs. The final score is based on
    summed ``buildFailures`` only, and is zero when the last observed run failed.
    """
    rev_scores: dict[str, dict[str, dict]] = {}

    for comp_rev in revisions:
        rev_scores[comp_rev] = {}
        for model in models:
            runs_for_model = task_runs.get(comp_rev, {}).get(model, {})
            if not runs_for_model:
                continue

            ordered = [runs_for_model[rn] for rn in sorted(runs_for_model.keys())]
            if not ordered:
                continue

            input_tokens = sum(tasks.safe_nonnegative_int(m.get("input_tokens", 0)) for m in ordered)
            output_tokens = sum(tasks.safe_nonnegative_int(m.get("output_tokens", 0)) for m in ordered)
            cached_input_tokens = sum(tasks.safe_nonnegative_int(m.get("cached_input_tokens", 0)) for m in ordered)
            build_failures = sum(tasks.safe_nonnegative_int(m.get("build", 0)) for m in ordered)
            parse_failures = sum(tasks.safe_nonnegative_int(m.get("parse", 0)) for m in ordered)
            apply_failures = sum(tasks.safe_nonnegative_int(m.get("apply", 0)) for m in ordered)
            api_retries = sum(tasks.safe_nonnegative_int(m.get("api", 0)) for m in ordered)
            turns = sum(tasks.safe_nonnegative_int(m.get("turns", 0)) for m in ordered)
            elapsed_ms = sum(tasks.safe_nonnegative_int(m.get("elapsed", 0)) for m in ordered)
            llm_ms = sum(tasks.safe_nonnegative_int(m.get("llm", 0)) for m in ordered)
            cost = sum(float(m.get("cost", 0.0)) for m in ordered)

            if cost == 0 and turns > 0:
                revision = comp_rev.split(":", 1)[1] if ":" in comp_rev else comp_rev
                print(
                    f"Warning: Cost is 0 for model '{model}' revision '{revision}'. Check pricing configuration.",
                    file=sys.stderr,
                )

            final_success = bool(ordered[-1].get("success", False))
            if final_success:
                score = 1.0 / math.log2(build_failures + 2)
            else:
                score = 0.0

            retries = build_failures + parse_failures + apply_failures + api_retries

            rev_scores[comp_rev][model] = {
                "score": score,
                "final_success": final_success,
                "runs_observed": len(ordered),
                "stop_reason": str(ordered[-1].get("stop_reason", "")),
                "cost": cost,
                "retries": retries,
                "build": build_failures,
                "parse": parse_failures,
                "apply": apply_failures,
                "api": api_retries,
                "turns": turns,
                "elapsed": elapsed_ms,
                "llm": llm_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_input_tokens,
            }

    return rev_scores


def _normalize_as_fraction_of_best(values: dict[str, float]) -> dict[str, float]:
    """
    Normalize values as a fraction of the best (minimum) value, scaled to [0, 100].

    For each model: score = 100 * (best_value / model_value)
    Best model gets 100%; others get proportionally lower scores.

    If all values are zero or missing, returns 50.0 for all keys.
    """
    valid_vals = [v for v in values.values() if v is not None and v > 0]
    if not valid_vals:
        return {k: 50.0 for k in values.keys()}

    best_val = min(valid_vals)
    if best_val <= 0:
        return {k: 50.0 for k in values.keys()}

    normalized: dict[str, float] = {}
    for k, v in values.items():
        if v is None or v <= 0:
            normalized[k] = 50.0
        else:
            normalized[k] = (best_val / v) * 100.0

    return normalized


def _plot(
    score_percent: dict[str, float],
    models: list[str],
    model_colors_global: dict[str, str],
    nolegend: bool = False,
) -> None:
    if not models:
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(1)

    score_vals = [score_percent.get(m, 0.0) for m in models]

    sorted_indices = sorted(range(len(models)), key=lambda i: score_vals[i], reverse=True)
    models_sorted = [models[i] for i in sorted_indices]
    scores_sorted = [score_vals[i] for i in sorted_indices]

    model_colors = [model_colors_global[m] for m in models_sorted]

    y = np.arange(len(models_sorted))
    fig, ax = plt.subplots(figsize=(10, max(6, 0.5 * len(models_sorted))))
    fig.canvas.mpl_connect('key_press_event', _on_key_press)

    rects = ax.barh(y, scores_sorted, color=model_colors)

    ax.set_xlabel("Score (%)")
    ax.set_title("Score by model" if not nolegend else "")
    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_name(m) for m in models_sorted])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))

    for i, _model in enumerate(models_sorted):
        width = rects[i].get_width()
        ax.text(
            width,
            rects[i].get_y() + rects[i].get_height() / 2,
            f" {width:.2f}%",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()


def _plot_by_revision(
    rev_scores: dict[str, dict[str, dict]], models: list[str], title: str, model_colors_global: dict[str, str], nolegend: bool = False,
) -> None:
    """
    Draw a horizontal grouped-bar chart with one group per revision (task) and
    one bar per model inside that group.

    Each revision occupies the integer Y coordinate; a simple fixed offset places
    the model bars within the slot, guaranteeing no overlap.
    """
    if not rev_scores:
        return

    # Revisions sorted from hardest (lowest average score) to easiest
    sorted_revs = sorted(
        rev_scores.items(),
        key=lambda item: sum(m["score"] for m in item[1].values()) / len(models),
    )

    revisions = [rev for rev, _ in sorted_revs]
    revision_labels = [
        (rev.split(":", 1)[1] if ":" in rev else rev)[:7] for rev in revisions
    ]

    n_revisions = len(revisions)
    n_models = len(models)

    group_height = 0.8              # vertical fraction used by all bars in a group
    bar_height = group_height / n_models

    fig_width = 12
    # 50 % taller for better readability
    fig_height = max(6, 0.675 * n_revisions)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.canvas.mpl_connect('key_press_event', _on_key_press)

    color_list = [model_colors_global[m] for m in models]
    cmap = mcolors.ListedColormap(color_list)

    # Decide label style: aggregate failures if *any* model contains "aider"
    aggregate_failures = any("aider" in m for m in models)

    for rev_index, (_rev, model_data) in enumerate(sorted_revs):
        base_y = rev_index
        for model_idx, model in enumerate(models):
            metrics = model_data.get(model)
            if not metrics:
                continue

            # Off-set each model into its own lane within the group
            offset = (-group_height / 2) + (model_idx + 0.5) * bar_height
            y = base_y + offset

            bar_width = metrics["score"]
            # Draw a tiny bar for failures so they are visible
            if bar_width == 0:
                bar_width = 0.001

            ax.barh(
                y,
                bar_width,
                # Leave a thin gap between bars in the same group
                height=bar_height * 0.8,
                color=cmap(model_idx),
                edgecolor="black",
            )

            # Build label depending on chosen failure-metric style
            cost = metrics["cost"]
            if math.isclose(metrics["score"], 1.0, abs_tol=1e-9):
                # Perfect score ⇒ no failure breakdown
                label_failures = ""
            else:
                if aggregate_failures:
                    label_failures = f"(R{metrics['retries']}) "
                else:
                    label_failures = (
                        f"(B{metrics['build']} "
                        f"P{metrics['parse']} "
                        f"A{metrics['apply']}) "
                    )
            label = (
                f"{metrics['score']:.2f} {label_failures}${cost:.3f}"
                if label_failures
                else f"${cost:.3f}"
            )
            text_x_pos = bar_width + 0.01
            # Special handling for 0-score runs to avoid overlapping text,
            # but don't show "FAIL" label to reduce clutter.
            if metrics["score"] == 0.0:
                text_x_pos = 0.01
                label = "" # Empty label for failed runs

            ax.text(
                text_x_pos,
                y,
                label,
                va="center",
                fontsize=7,
                clip_on=False,
            )

    ax.set_yticks(range(n_revisions))
    ax.set_yticklabels(revision_labels)
    ax.invert_yaxis()  # hardest tasks at the top
    ax.set_xlabel("Score")
    ax.set_title(title if not nolegend else "")
    ax.set_xlim(0, 1)

    if not nolegend:
        legend_handles = [
            plt.matplotlib.patches.Patch(color=cmap(i), label=_pretty_name(model))
            for i, model in enumerate(models)
        ]
        ax.legend(handles=legend_handles, title="Models", loc="upper right", fontsize=8)

    plt.tight_layout()


def _plot_llm_by_revision(
    rev_scores: dict[str, dict[str, dict]],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str],
    nolegend: bool = False,
) -> None:
    """
    Draw a horizontal grouped-bar chart with one group per revision (task) and one
    bar per model inside that group, where the x-axis is normalized LLM runtime.

    Uses aggregated per-task llmMillis (summed across observed runs), normalized
    by each model's fastest successful task runtime.
    """
    if not rev_scores:
        return

    min_success_llm_ms: dict[str, float] = {}
    for model in models:
        llms: list[float] = []
        for _rev, model_data in rev_scores.items():
            metrics = model_data.get(model)
            if not metrics:
                continue
            if metrics.get("final_success", False) and metrics.get("llm", 0) > 0:
                llms.append(float(metrics["llm"]))
        if llms:
            min_success_llm_ms[model] = min(llms)

    active_models = [m for m in models if m in min_success_llm_ms]
    if not active_models:
        return

    rev_median_ratio: dict[str, float] = {}
    for comp_rev, model_data in rev_scores.items():
        ratios: list[float] = []
        for m in active_models:
            metrics = model_data.get(m)
            total_llm_ms = float(metrics.get("llm", 0.0)) if metrics else 0.0
            denom = float(min_success_llm_ms[m])
            ratio = (total_llm_ms / denom) if denom > 0 else 0.0
            ratios.append(ratio)
        rev_median_ratio[comp_rev] = float(np.median(ratios)) if ratios else 0.0

    sorted_revs = sorted(
        rev_scores.items(),
        key=lambda item: rev_median_ratio[item[0]],
        reverse=True,
    )

    revisions = [comp_rev for comp_rev, _ in sorted_revs]
    # Show only the task hash (drop project if present) and truncate
    revision_labels = [
        (comp_rev.split(":", 1)[1] if ":" in comp_rev else comp_rev)[:7]
        for comp_rev in revisions
    ]

    n_revisions = len(revisions)
    n_models = len(active_models)

    group_height = 0.8
    bar_height = group_height / n_models

    fig_width = 12
    fig_height = max(6, 0.675 * n_revisions)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.canvas.mpl_connect('key_press_event', _on_key_press)

    color_list = [model_colors_global[m] for m in active_models]
    cmap = mcolors.ListedColormap(color_list)

    max_ratio = 1.0

    for rev_index, (_rev, model_data) in enumerate(sorted_revs):
        base_y = rev_index
        for model_idx, model in enumerate(active_models):
            metrics = model_data.get(model)
            total_llm_ms = float(metrics.get("llm", 0.0)) if metrics else 0.0

            offset = (-group_height / 2) + (model_idx + 0.5) * bar_height
            y = base_y + offset

            denom = float(min_success_llm_ms[model])
            if denom > 0 and total_llm_ms > 0:
                ratio = total_llm_ms / denom
                bar_width = ratio if ratio > 0 else 0.001
                max_ratio = max(max_ratio, ratio)
                label = f"{ratio:.2f}×"
                text_x_pos = max(bar_width + 0.02, 0.02)
            else:
                bar_width = 0.001
                label = ""
                text_x_pos = 0.01

            ax.barh(
                y,
                bar_width,
                height=bar_height * 0.8,
                color=cmap(model_idx),
                edgecolor="black",
            )

            if label:
                ax.text(
                    text_x_pos,
                    y,
                    label,
                    va="center",
                    fontsize=7,
                    clip_on=False,
                )

    ax.set_yticks(range(n_revisions))
    ax.set_yticklabels(revision_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized LLM Runtime (× min per model)")
    ax.set_title(title if not nolegend else "")
    ax.set_xlim(0, max_ratio * 1.05 if max_ratio > 0 else 1.0)

    if not nolegend:
        legend_handles = [
            plt.matplotlib.patches.Patch(color=cmap(i), label=_pretty_name(model))
            for i, model in enumerate(active_models)
        ]
        ax.legend(handles=legend_handles, title="Models", loc="upper right", fontsize=8)

    plt.tight_layout()


def _format_seconds(seconds: float) -> str:
    """Convert seconds to a concise "XmYs" representation."""
    if seconds >= 60:
        minutes = int(seconds) // 60
        secs = int(round(seconds - minutes * 60))
        return f"{minutes}m{secs}s"
    return f"{seconds:.1f}s" if seconds < 10 else f"{int(round(seconds))}s"


def _plot_success_scatter(
    x_data: dict[str, float],
    y_data: dict[str, float],
    models: list[str],
    title: str,
    x_label: str,
    x_formatter: Callable,
    x_axis_formatter: Callable | None = None,
    model_colors_global: dict[str, str] | None = None,
) -> None:
    """
    Scatter plot for successful tasks only, with filled circles and model names displayed
    above the value to the right of each dot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.mpl_connect("key_press_event", _on_key_press)

    n_models = len(models)
    if model_colors_global is None:
        color_list = _get_distinct_colors(n_models)
    else:
        color_list = [model_colors_global[m] for m in models]
    cmap = mcolors.ListedColormap(color_list)

    for i, model in enumerate(models):
        x_val = x_data.get(model, 0.0)
        y_val = y_data.get(model, 0.0)

        # Filled circle for successful tasks (larger size)
        ax.scatter(x_val, y_val, color=cmap(i), s=200, zorder=3)

        # Add two-line label next to the dot: model name above, value below
        model_name = _pretty_name(model)
        value_text = x_formatter(x_val)
        label_text = f"   {model_name}\n   {value_text}"
        ax.text(x_val, y_val, label_text, fontsize=9, va="center", clip_on=False)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    if x_axis_formatter:
        ax.xaxis.set_major_formatter(x_axis_formatter)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)

    plt.tight_layout()


def _plot_latency_scatter(
    estimated_latency: dict[str, float],
    y_data: dict[str, float],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str] | None = None,
    nolegend: bool = False,
) -> None:
    """
    Scatter plot displaying estimated solve-all latency.
    """
    _plot_success_scatter(
        x_data=estimated_latency,
        y_data=y_data,
        models=models,
        title=title if not nolegend else "",
        x_label="LLM Latency (s)",
        x_formatter=_format_seconds,
        x_axis_formatter=mticker.FuncFormatter(lambda x, _p: f"{x:,.0f}"),
        model_colors_global=model_colors_global,
    )


def _plot_spend_scatter(
    estimated_spend: dict[str, float],
    y_data: dict[str, float],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str] | None = None,
    nolegend: bool = False,
) -> None:
    """
    Scatter plot displaying estimated solve-all spend.
    """
    _plot_success_scatter(
        x_data=estimated_spend,
        y_data=y_data,
        models=models,
        title=title if not nolegend else "",
        x_label="Spend ($)",
        x_formatter=lambda cost: f"${cost:.2f}",
        model_colors_global=model_colors_global,
    )


# ---------- Task-length bucketing utilities ----------


def _bucket_for_tokens(tokens: int) -> str:
    """
    Determine the power-of-two bucket label for the given *tokens* count.

    The bucket is defined by::

        lower = 2**k
        upper = 2**(k + 1) - 1

    such that ``lower <= tokens <= upper`` for some integer *k* ≥ 0.
    """
    if tokens <= 0:
        return "0"
    k = int(math.floor(math.log2(tokens)))
    lower = 1 << k
    upper = (1 << (k + 1)) - 1
    return f"{lower}-{upper}"


def _build_bucket_scores(
    rev_scores: dict[str, dict[str, dict]],
    tokens_by_rev: dict[str, int],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """
    Aggregate per-revision scores into task-length buckets.
    Returns (bucket_scores, task_counts_in_bucket).
    """
    bucket_scores: dict[str, dict[str, float]] = {}
    task_counts: dict[str, int] = {}
    for rev, model_scores in rev_scores.items():
        tokens = tokens_by_rev.get(rev)
        if tokens is None:
            continue
        bucket = _bucket_for_tokens(tokens)
        task_counts[bucket] = task_counts.get(bucket, 0) + 1
        for model, metrics in model_scores.items():
            bucket_scores.setdefault(bucket, {}).setdefault(model, 0.0)
            bucket_scores[bucket][model] += metrics["score"]
    return bucket_scores, task_counts


def _language_output_key(language: str) -> str:
    normalized = "".join(
        char if (char.isalnum() and char.isascii()) else "_"
        for char in language.strip().lower()
    )
    normalized = normalized.strip("_")
    if not normalized:
        normalized = "unknown"
    return f"language_{normalized}"


def _collect_language_totals(
    rev_scores: dict[str, dict[str, dict]],
    valid_revisions: list[str],
    models: list[str],
    revision_languages: dict[str, str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """
    Aggregate score/pass/cost/performance stats per model for each language.
    """
    language_totals: dict[str, dict[str, dict[str, float | int]]] = {}
    for lang in {"all", *revision_languages.values()}:
        language_totals[lang] = {}

    for comp_rev in valid_revisions:
        language = revision_languages.get(comp_rev, "unknown")
        model_scores = rev_scores.get(comp_rev, {})
        for model in models:
            metrics = model_scores.get(model)
            if not metrics:
                continue
            score = float(metrics.get("score", 0.0))
            for scope in ("all", language):
                scope_totals = language_totals.setdefault(scope, {})
                model_totals = scope_totals.setdefault(
                    model,
                    {
                        "task_count": 0,
                        "score_sum": 0.0,
                        "pass_count": 0,
                        "build_failures": 0,
                        "spend_sum": 0.0,
                        "latency_seconds_sum": 0.0,
                        "input_tokens_sum": 0,
                        "output_tokens_sum": 0,
                        "cached_input_tokens_sum": 0,
                        "turns_sum": 0,
                    },
                )
                model_totals["task_count"] += 1
                model_totals["score_sum"] += score
                if bool(metrics.get("final_success", False)):
                    model_totals["pass_count"] += 1
                model_totals["build_failures"] += int(metrics.get("build", 0))
                model_totals["spend_sum"] += tasks.safe_nonnegative_float(
                    metrics.get("spend", 0.0),
                )
                model_totals["latency_seconds_sum"] += tasks.safe_nonnegative_float(
                    metrics.get("latency_seconds", 0.0),
                )
                model_totals["input_tokens_sum"] += tasks.safe_nonnegative_int(
                    metrics.get("input_tokens", 0),
                )
                model_totals["output_tokens_sum"] += tasks.safe_nonnegative_int(
                    metrics.get("output_tokens", 0),
                )
                model_totals["cached_input_tokens_sum"] += tasks.safe_nonnegative_int(
                    metrics.get("cached_input_tokens", 0),
                )
                model_totals["turns_sum"] += tasks.safe_nonnegative_int(
                    metrics.get("turns", 0),
                )
    return language_totals


def _collect_language_bucket_totals(
    rev_scores: dict[str, dict[str, dict]],
    valid_revisions: list[str],
    tokens_by_rev: dict[str, int],
    models: list[str],
    revision_languages: dict[str, str],
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    """
    Aggregate score/pass counts by bucket and language for each model.
    Returns: bucket -> language -> model -> agg.
    """
    bucket_totals: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for comp_rev in valid_revisions:
        tokens = tokens_by_rev.get(comp_rev)
        if tokens is None:
            continue
        bucket = _bucket_for_tokens(tokens)
        language = revision_languages.get(comp_rev, "unknown")
        model_scores = rev_scores.get(comp_rev, {})
        for model in model_scores:
            metrics = model_scores.get(model)
            if not metrics or model not in models:
                continue
            score = float(metrics.get("score", 0.0))
            for scope in ("all", language):
                bucket_scope = bucket_totals.setdefault(bucket, {}).setdefault(scope, {})
                scope_totals = bucket_scope.setdefault(
                    model,
                    {
                        "task_count": 0,
                        "score_sum": 0.0,
                        "pass_count": 0,
                    },
                )
                scope_totals["task_count"] += 1
                scope_totals["score_sum"] += score
                if bool(metrics.get("final_success", False)):
                    scope_totals["pass_count"] += 1
    return bucket_totals


def _build_language_payload(
    stats: dict[str, float | int] | None,
    *,
    task_count_key: str,
    include_spend: bool = False,
    include_latency: bool = False,
    include_build_failures: bool = False,
    include_input_tokens: bool = False,
    include_output_tokens: bool = False,
    include_cached_input_tokens: bool = False,
    include_turns: bool = False,
) -> dict[str, float | int]:
    task_count = int(stats.get("task_count", 0)) if stats else 0
    pass_count = int(stats.get("pass_count", 0)) if stats else 0
    score_sum = float(stats.get("score_sum", 0.0)) if stats else 0.0
    payload: dict[str, float | int] = {
        task_count_key: task_count,
        "pass_count": pass_count,
        "pass_rate": (float(pass_count) / task_count * 100.0) if task_count else 0.0,
        "score_percent": (score_sum / task_count * 100.0) if task_count else 0.0,
    }
    if include_build_failures:
        payload["build_failures"] = int(stats.get("build_failures", 0)) if stats else 0
    if include_spend:
        payload["spend"] = float(stats.get("spend_sum", 0.0)) if stats else 0.0
    if include_latency:
        payload["latency_seconds"] = (
            float(stats.get("latency_seconds_sum", 0.0)) if stats else 0.0
        )
    if include_input_tokens:
        payload["input_tokens"] = float(stats.get("input_tokens_sum", 0)) if stats else 0.0
    if include_output_tokens:
        payload["output_tokens"] = float(stats.get("output_tokens_sum", 0)) if stats else 0.0
    if include_cached_input_tokens:
        payload["cached_input_tokens"] = (
            float(stats.get("cached_input_tokens_sum", 0)) if stats else 0.0
        )
    if include_turns:
        payload["turns"] = int(stats.get("turns_sum", 0)) if stats else 0
    return payload


def _attach_language_payloads(
    row: dict[str, object],
    *,
    language_totals: dict[str, dict[str, dict[str, float | int]]],
    language_order: list[str],
    model: str,
    task_count_key: str = "task_count",
    include_spend: bool = False,
    include_latency: bool = False,
    include_build_failures: bool = False,
    include_input_tokens: bool = False,
    include_output_tokens: bool = False,
    include_cached_input_tokens: bool = False,
    include_turns: bool = False,
) -> None:
    """
    Add language_* entries to a row dictionary for one model.
    """
    for language in language_order:
        key = _language_output_key(language)
        row[key] = _build_language_payload(
            language_totals.get(language, {}).get(model),
            task_count_key=task_count_key,
            include_spend=include_spend,
            include_latency=include_latency,
            include_build_failures=include_build_failures,
            include_input_tokens=include_input_tokens,
            include_output_tokens=include_output_tokens,
            include_cached_input_tokens=include_cached_input_tokens,
            include_turns=include_turns,
        )


def _plot_radar(
    normalized_score: dict[str, float],
    normalized_speed: dict[str, float],
    normalized_price: dict[str, float],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str],
    nolegend: bool = False,
) -> None:
    """
    Draw a radar plot (spider chart) with three axes (Score, Speed, Value),
    each normalized to [0, 100] where 100% = best on each dimension.

    Each model is represented as a filled polygon connecting its three normalized values.
    """
    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.canvas.mpl_connect("key_press_event", _on_key_press)

    # Three axes for the radar
    axes_names = ["Score", "Speed", "Value"]
    num_axes = len(axes_names)

    # Angle for each axis (in radians)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    # Close the plot by repeating the first angle
    angles += angles[:1]

    cmap = mcolors.ListedColormap([model_colors_global[m] for m in models])

    for i, model in enumerate(models):
        values = [
            normalized_score.get(model, 50.0),
            normalized_speed.get(model, 50.0),
            normalized_price.get(model, 50.0),
        ]
        # Close the polygon by repeating the first value
        values += values[:1]

        # Draw the filled polygon with semi-transparency
        ax.plot(angles, values, 'o-', linewidth=2, label=_pretty_name(model), color=cmap(i))
        ax.fill(angles, values, alpha=0.15, color=cmap(i))

    # Set axis labels and limits
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_names, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{int(y)}" for y in np.arange(0, 101, 20)], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_title(title if not nolegend else "", fontsize=14, pad=20)
    if not nolegend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()


def _plot_by_bucket(
    bucket_scores: dict[str, dict[str, float]],
    bucket_task_counts: dict[str, int],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str],
    nolegend: bool = False,
) -> None:
    """
    Draw a grouped horizontal bar chart, with one group per token-length bucket.
    """
    if not bucket_scores:
        return

    def _bucket_key(label: str) -> int:
        lower_part = label.split("-")[0]
        try:
            return int(lower_part)
        except ValueError:
            return 0

    sorted_buckets = sorted(bucket_scores.keys(), key=_bucket_key)

    n_buckets = len(sorted_buckets)
    n_models = len(models)

    group_height = 0.8
    bar_height = group_height / n_models

    fig_height = max(6, 0.675 * n_buckets)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.canvas.mpl_connect("key_press_event", _on_key_press)

    color_list = [model_colors_global[m] for m in models]
    cmap = mcolors.ListedColormap(color_list)

    for b_idx, bucket_label in enumerate(sorted_buckets):
        base_y = b_idx
        task_count_in_bucket = bucket_task_counts.get(bucket_label, 0)
        if task_count_in_bucket == 0:
            continue

        for m_idx, model in enumerate(models):
            sum_score_val = bucket_scores.get(bucket_label, {}).get(model, 0.0)
            normalized_score = (sum_score_val / task_count_in_bucket) * 100

            offset = (-group_height / 2) + (m_idx + 0.5) * bar_height
            y = base_y + offset

            bar_width = normalized_score if normalized_score > 0 else 0.001
            ax.barh(
                y,
                bar_width,
                height=bar_height * 0.8,
                color=cmap(m_idx),
                edgecolor="black",
            )
            if normalized_score:
                ax.text(
                    bar_width + 0.01,
                    y,
                    f"{normalized_score:.2f}%",
                    va="center",
                    fontsize=7,
                    clip_on=False,
                )

    ax.set_yticks(range(n_buckets))
    ax.set_yticklabels([f"{b} ({bucket_task_counts.get(b, 0)})" for b in sorted_buckets])
    ax.invert_yaxis()
    ax.set_xlabel("Score (%)")
    ax.set_title(title if not nolegend else "")
    ax.set_xlim(0, 100) # Set x-limit to 100%

    if not nolegend:
        legend_handles = [
            plt.matplotlib.patches.Patch(color=cmap(i), label=model)
            for i, model in enumerate(models)
        ]
        ax.legend(handles=legend_handles, title="Models", loc="upper right", fontsize=8)

    plt.tight_layout()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(file_path: Path, rows: Iterable[dict]) -> None:
    _ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------- ASCII rendering helpers -------------------

def _has_display() -> bool:
    """
    Heuristic to decide if we should render matplotlib figures:

    - True for Jupyter inline backends
    - True for known interactive GUI backends (MacOSX/Qt/Tk/GTK/WX)
    - True on Windows (historical behavior in this tool)
    - Otherwise, require a display environment (DISPLAY/WAYLAND_DISPLAY)
    """
    try:
        backend = (plt.get_backend() or "").lower()
    except Exception:
        backend = ""

    # Known interactive backends
    interactive_tokens = {
        "inline",   # jupyter
        "macosx",   # native macOS
        "qtagg", "qt5agg", "qt6agg",
        "tkagg",
        "gtk3agg", "gtk4agg",
        "wxagg",
    }

    if any(token in backend for token in interactive_tokens):
        return True

    # Preserve historical behavior on Windows
    if sys.platform.startswith("win"):
        return True

    # Fallback: require a display (common on Linux headless)
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _bar_str(value: float, max_value: float, width: int, formatter: Callable[[float], str]) -> str:
    """
    Build a horizontal ASCII bar for a given value in [0, max_value].
    """
    if not math.isfinite(value) or not math.isfinite(max_value) or max_value <= 0:
        filled = 0
    else:
        filled = int(round((value / max_value) * width))
    bar = "#" * max(0, min(filled, width))
    return f"{bar} {formatter(value)}"


def _print_ascii_single_series(
    title: str,
    labels: list[str],
    values: dict[str, float],
    width: int,
    formatter: Callable[[float], str],
    model_colors: dict[str, str] | None = None,
    color_lines: bool = False,
) -> None:
    """
    Print one bar per label.
    """
    print('# ' + title)
    max_label = max((len(_pretty_name(label)) for label in labels), default=0)
    max_value = max([values.get(label, 0.0) for label in labels] + [0.0])
    for label in labels:
        color = model_colors.get(label) if model_colors else None
        display = _pretty_name(label).ljust(max_label)
        v = float(values.get(label, 0.0))
        bar = _bar_str(v, max_value, width, formatter)
        display = _colorize(display, color, enabled=color_lines)
        bar = _colorize(bar, color, enabled=color_lines)
        print(f"{display} | {bar}")
    print("")


def _print_ascii_bucketed(
    title: str,
    bucket_scores: dict[str, dict[str, float]],
    bucket_task_counts: dict[str, int],
    models: list[str],
    width: int,
    model_colors: dict[str, str] | None = None,
    color_lines: bool = False,
) -> None:
    """
    Print grouped bars per token-length bucket (normalized to percentages).
    """
    if not bucket_scores:
        return

    def _bucket_key(label: str) -> int:
        lower_part = label.split("-")[0]
        try:
            return int(lower_part)
        except ValueError:
            return 0

    sorted_buckets = sorted(bucket_scores.keys(), key=_bucket_key)
    print(title)
    for b in sorted_buckets:
        tasks_in_bucket = int(bucket_task_counts.get(b, 0))
        print(f"- {b} ({tasks_in_bucket})")
        if tasks_in_bucket == 0:
            continue
        # Build normalized percentage per model for this bucket
        vals = {
            m: (float(bucket_scores.get(b, {}).get(m, 0.0)) / tasks_in_bucket) * 100.0
            for m in models
        }
        max_label = max((len(_pretty_name(m)) for m in models), default=0)
        for m in models:
            color = model_colors.get(m) if model_colors else None
            display = _pretty_name(m).ljust(max_label)
            v = vals.get(m, 0.0)
            # Normalize percentages to the 0-100 range so that 100% fills the bar.
            bar = _bar_str(v, 100.0, width, lambda x: f"{x:.2f}%")
            display = _colorize(display, color, enabled=color_lines)
            bar = _colorize(bar, color, enabled=color_lines)
            print(f"  {display} | {bar}")
    print("")


def _run_number_from_dir_name(project_name: str, run_dir: Path) -> int | None:
    """Extract numeric run suffix from a run directory name like '<project><N>'."""
    _discovered_project, suffix = tasks.parse_project_and_run(run_dir)
    if _discovered_project != project_name or suffix is None:
        return None
    return suffix


def _log_discarded_hashes(discarded: dict[str, set[str]], label: str = "task hashes") -> None:
    """Log discarded items and reasons before scoring begins."""
    if not discarded:
        return
    print(f"# Discarded {label} (before scoring):", file=sys.stderr)
    for comp_rev in sorted(discarded.keys()):
        reasons = ", ".join(sorted(discarded[comp_rev]))
        print(f"{comp_rev} [{reasons}]", file=sys.stderr)
    print("", file=sys.stderr)


def main() -> None:
    (
        run_info,
        models_order_from_cli,
        exclude_models_filter,
        filter_task_difficulty,
        filter_revisions,
        projects,
        tasksize,
        force_text,
        charts_to_render,
        nolegend,
        exclude_missing,
        json_export_dir,
        no_estimate,
    ) = _parse_args()

    results_dir = run_info[0][1].parent
    _configure_model_metadata_path(results_dir)
    _configure_project_metadata_path(results_dir)

    text_mode = bool(force_text) or not _has_display()

    models_filter: set[str] | None = None
    if models_order_from_cli is not None:
        models_filter = {m for m in models_order_from_cli if m != "_"}

    run_results: list[tuple[str, int, Path, dict[str, dict[str, dict]]]] = []
    for project_name, run_dir in run_info:
        run_number = _run_number_from_dir_name(project_name, run_dir)
        if run_number is None:
            print(f"Skipping run {run_dir.name}: could not parse numeric run suffix", file=sys.stderr)
            continue
        try:
            rev_scores = _aggregate_run(
                run_dir,
                run_number,
                models_filter,
                exclude_models_filter,
                tasksize,
                filter_revisions,
            )
            run_results.append((project_name, run_number, run_dir, rev_scores))
        except ValueError as exc:
            print(f"Skipping run {run_dir.name} due to error: {exc}", file=sys.stderr)
            continue

    if not run_results:
        print("No valid run data found.", file=sys.stderr)
        sys.exit(1)

    requested_projects = {project_name for project_name, _run_number, _run_dir, _rev_scores in run_results}
    missing_projects = sorted(
        requested_projects - set(_PROJECT_LANGUAGES.keys())
    )
    if missing_projects:
        print(
            "ERROR: missing language metadata for projects: "
            + ", ".join(missing_projects),
            file=sys.stderr,
        )
        print(
            f"Update {_PROJECT_METADATA_PATH} to include language entries for these projects.",
            file=sys.stderr,
        )
        sys.exit(1)

    export_dir = None
    if json_export_dir is not None:
        export_dir = run_results[0][2].parent / "webapp-json" / json_export_dir

    all_models: set[str] = set()
    for _project_name, _run_number, _run_dir, rev_scores in run_results:
        for _rev, model_data in rev_scores.items():
            all_models.update(model_data.keys())

    if models_order_from_cli is not None:
        models = [m for m in models_order_from_cli if m != "_" and m in all_models]
        remaining = sorted(all_models - set(models))
        models.extend(remaining)
    else:
        models = sorted(all_models)

    if not models:
        print("No models found after filtering.", file=sys.stderr)
        sys.exit(1)

    model_colors_global: dict[str, str] = {}
    if models_order_from_cli is not None:
        color_palette = _get_distinct_colors(len(models_order_from_cli))
        color_index = 0
        for entry in models_order_from_cli:
            if entry == "_":
                color_index += 1
                continue
            if entry in all_models:
                model_colors_global[entry] = color_palette[color_index]
            color_index += 1
        for model in models:
            if model not in model_colors_global:
                model_colors_global[model] = color_palette[color_index % len(color_palette)]
                color_index += 1
    else:
        color_palette = _get_distinct_colors(len(models))
        for i, model in enumerate(models):
            model_colors_global[model] = color_palette[i]

    # task_runs[composite_revision][model][run_number] = run_metrics
    task_runs: dict[str, dict[str, dict[int, dict]]] = {}
    for project_name, run_number, _run_dir, rev_scores in run_results:
        for rev, model_data in rev_scores.items():
            comp_rev = f"{project_name}:{rev}"
            for model, metrics in model_data.items():
                if model not in models:
                    continue
                task_runs.setdefault(comp_rev, {}).setdefault(model, {})[run_number] = metrics

    if not task_runs:
        print("No task/model run data found after filtering.", file=sys.stderr)
        sys.exit(1)

    max_run_number = max(run_number for _project_name, run_number, _run_dir, _rev_scores in run_results)
    if exclude_missing == "models":
        models, discarded = _validate_models(task_runs, models, max_run_number)
        _log_discarded_hashes(discarded, label="models")
        if not models:
            print("No models with complete task coverage remain after validation.", file=sys.stderr)
            sys.exit(1)
        valid_revisions = sorted(task_runs.keys())
    else:
        valid_revisions, discarded = _validate_task_hashes(task_runs, models, max_run_number)
        _log_discarded_hashes(discarded)

    if not valid_revisions:
        print("No valid task hashes remain after validation.", file=sys.stderr)
        sys.exit(1)

    rev_scores = _aggregate_task_runs(task_runs, valid_revisions, models)

    n_revisions = len(valid_revisions)
    n_runs = len(run_results)
    project_label = ", ".join(projects)

    sum_scores = {m: 0.0 for m in models}
    pass_counts: dict[str, int] = {m: 0 for m in models}
    total_build_failures: dict[str, int] = {m: 0 for m in models}
    for comp_rev in valid_revisions:
        model_data = rev_scores.get(comp_rev, {})
        for model in models:
            metrics = model_data.get(model)
            if not metrics:
                continue
            sum_scores[model] += float(metrics.get("score", 0.0))
            if metrics.get("final_success", False):
                pass_counts[model] += 1
                total_build_failures[model] += int(metrics.get("build", 0))

    score_percent = {
        m: (sum_scores[m] / n_revisions) * 100.0 if n_revisions else 0.0
        for m in models
    }

    model_groups: dict[str, list[str]] = {}
    for model_alias in models:
        raw_model_name = MODEL_MAPPING.get(model_alias, model_alias)
        model_groups.setdefault(raw_model_name, []).append(model_alias)
    sorted_group_names = sorted(
        model_groups.keys(),
        key=lambda g: max(score_percent.get(alias, 0.0) for alias in model_groups[g]),
        reverse=True,
    )
    models = [
        alias
        for group_name in sorted_group_names
        for alias in sorted(
            model_groups[group_name],
            key=lambda m: score_percent.get(m, 0.0),
            reverse=True,
        )
    ]

    tokens_by_rev = _load_tokens_by_revision(projects, run_results, valid_revisions)
    if no_estimate:
        estimated_solve_all_spend: dict[str, float] = {m: 0.0 for m in models}
        estimated_task_cost: dict[tuple[str, str], float] = {}
        estimated_solve_all_latency_seconds: dict[str, float] = {m: 0.0 for m in models}
        estimated_task_latency_seconds: dict[tuple[str, str], float] = {}
        calibration_input_factors = {}
        calibration_output_factors = {}
        calibration_latency_factors = {}
        slice_diagnostics = {}
        per_model_slice_error = {}
        for comp_rev in valid_revisions:
            for model in models:
                metrics = rev_scores.get(comp_rev, {}).get(model)
                if not metrics:
                    continue
                task_key = (comp_rev, model)
                estimated_task_cost[task_key] = _compute_cost_from_tokens(
                    model=model,
                    revision=comp_rev,
                    input_tokens=tasks.safe_nonnegative_float(metrics.get("input_tokens", 0.0)),
                    output_tokens=tasks.safe_nonnegative_float(metrics.get("output_tokens", 0.0)),
                    cached_input_tokens=tasks.safe_nonnegative_float(metrics.get("cached_input_tokens", 0.0)),
                )
                estimated_solve_all_spend[model] += estimated_task_cost[task_key]
                llm_ms = tasks.safe_nonnegative_float(metrics.get("llm", 0.0))
                if llm_ms == 0.0:
                    llm_ms = tasks.safe_nonnegative_float(metrics.get("elapsed", 0.0))
                estimated_task_latency_seconds[task_key] = llm_ms / 1000.0
                estimated_solve_all_latency_seconds[model] += estimated_task_latency_seconds[task_key]
    else:
        (
            estimated_solve_all_spend,
            estimated_task_cost,
            estimated_solve_all_latency_seconds,
            estimated_task_latency_seconds,
            calibration_input_factors,
            calibration_output_factors,
            calibration_latency_factors,
            slice_diagnostics,
            per_model_slice_error,
        ) = _compute_estimated_solve_all_metrics(
            rev_scores=rev_scores,
            models=models,
            revisions=valid_revisions,
            tokens_by_rev=tokens_by_rev,
            filter_task_difficulty=filter_task_difficulty,
        )
    estimated_solve_all_spend_all_models = {
        m: float(estimated_solve_all_spend.get(m, float("nan"))) for m in models
    }
    estimated_solve_all_latency_all_models = {
        m: float(estimated_solve_all_latency_seconds.get(m, float("nan"))) for m in models
    }
    for comp_rev in valid_revisions:
        for model in models:
            metrics = rev_scores.get(comp_rev, {}).get(model)
            if not metrics:
                continue
            task_key = (comp_rev, model)
            if task_key in estimated_task_cost:
                estimated_cost = float(estimated_task_cost[task_key])
                metrics["spend"] = estimated_cost
                metrics["cost"] = estimated_cost
            else:
                metrics["spend"] = float("nan")

            task_latency_key = (comp_rev, model)
            if task_latency_key in estimated_task_latency_seconds:
                metrics["latency_seconds"] = float(
                    estimated_task_latency_seconds[task_latency_key]
                )
            else:
                metrics["latency_seconds"] = float("nan")

    if text_mode:
        use_color = _ansi_color_supported()
        _print_ascii_single_series(
            title="Score" if not nolegend else "",
            labels=models,
            values=score_percent,
            width=50,
            formatter=lambda v: f"{v:.2f}%",
            model_colors=model_colors_global,
            color_lines=use_color,
        )

        print("# Pass Count (build failures for successful tasks)" if not nolegend else "")
        max_label = max((len(_pretty_name(m)) for m in models), default=0)
        bar_width = 50
        for m in models:
            color = model_colors_global.get(m)
            display = _pretty_name(m).ljust(max_label)
            passes = int(pass_counts[m])
            total = int(n_revisions)
            builds = int(total_build_failures[m])
            pass_rate = (passes / total * 100.0) if total > 0 else 0.0
            filled = int(round((pass_rate / 100.0) * bar_width))
            bar = "#" * max(0, min(filled, bar_width))
            line = (
                f"{display} | {bar:<{bar_width}} {passes:>2}/{total:<2} ({builds:>2}) {pass_rate:.1f}%"
            )
            print(_colorize(line, color, enabled=use_color))
        print("")
    else:
        if "mainscore" in charts_to_render:
            _plot(
                score_percent=score_percent,
                models=models,
                model_colors_global=model_colors_global,
                nolegend=nolegend,
            )
        if "per_task" in charts_to_render:
            _plot_by_revision(
                rev_scores,
                models,
                title=f"Per-Task Scores (Projects: {project_label}, {n_runs} runs)" if not nolegend else "",
                model_colors_global=model_colors_global,
                nolegend=nolegend,
            )
        if "llm_runtime" in charts_to_render:
            _plot_llm_by_revision(
                rev_scores,
                models,
                title="Per-Task LLM Runtime (Normalized)",
                model_colors_global=model_colors_global,
                nolegend=nolegend,
            )

    if text_mode:
        cost_title = (
            f"Total Cost (Projects: {project_label}, {n_runs} runs)"
            if no_estimate
            else f"Estimated Solve-All Cost (Projects: {project_label}, {n_runs} runs)"
        )
        latency_title = (
            f"Total Latency (Projects: {project_label}, {n_runs} runs)"
            if no_estimate
            else f"Estimated Solve-All Latency (Projects: {project_label}, {n_runs} runs)"
        )
        _print_ascii_single_series(
            title=cost_title if not nolegend else "",
            labels=models,
            values=estimated_solve_all_spend_all_models,
            width=50,
            formatter=lambda c: f"${c:.2f}",
            model_colors=model_colors_global,
            color_lines=use_color,
        )
        _print_ascii_single_series(
            title=latency_title if not nolegend else "",
            labels=models,
            values=estimated_solve_all_latency_all_models,
            width=50,
            formatter=_format_seconds,
            model_colors=model_colors_global,
            color_lines=use_color,
        )
        if not no_estimate:
            _print_calibration_diagnostics(
                models=models,
                input_factors=calibration_input_factors,
                output_factors=calibration_output_factors,
                latency_factors=calibration_latency_factors,
                per_model_slice_error=per_model_slice_error,
                slice_diagnostics=slice_diagnostics,
            )
    else:
        if "score_v_spend" in charts_to_render:
            spend_title = (
                "Score vs. Total Cost"
                if no_estimate
                else "Score vs. Estimated Solve-All Cost"
            )
            _plot_spend_scatter(
                estimated_spend=estimated_solve_all_spend_all_models,
                y_data=score_percent,
                models=models,
                title=spend_title if not nolegend else "",
                model_colors_global=model_colors_global,
                nolegend=nolegend,
            )
        if "score_v_latency" in charts_to_render:
            latency_title = (
                "Score vs. Total Latency"
                if no_estimate
                else "Score vs. Estimated Solve-All LLM Latency"
            )
            _plot_latency_scatter(
                estimated_latency=estimated_solve_all_latency_all_models,
                y_data=score_percent,
                models=models,
                title=latency_title if not nolegend else "",
                model_colors_global=model_colors_global,
                nolegend=nolegend,
            )
        if "score_speed_price" in charts_to_render:
            norm_speed = _normalize_as_fraction_of_best(estimated_solve_all_latency_all_models)
            norm_price = _normalize_as_fraction_of_best(estimated_solve_all_spend_all_models)
            _plot_radar(
                normalized_score=score_percent,
                normalized_speed=norm_speed,
                normalized_price=norm_price,
                models=models,
                title="Score vs. Speed vs. Value" if not nolegend else "",
                model_colors_global=model_colors_global,
                nolegend=nolegend,
            )

    bucket_scores: dict[str, dict[str, float]] = {}
    bucket_counts: dict[str, int] = {}
    bucket_language_totals: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    revision_languages = {
        comp_rev: _project_language(comp_rev.split(":", 1)[0])
        if ":" in comp_rev
        else _project_language("")
        for comp_rev in valid_revisions
    }
    language_order = ["all", *sorted(set(revision_languages.values()))]
    language_totals = _collect_language_totals(
        rev_scores=rev_scores,
        valid_revisions=valid_revisions,
        models=models,
        revision_languages=revision_languages,
    )

    if tokens_by_rev:
        bucket_scores, bucket_counts = _build_bucket_scores(rev_scores, tokens_by_rev)
        bucket_language_totals = _collect_language_bucket_totals(
            rev_scores=rev_scores,
            valid_revisions=valid_revisions,
            tokens_by_rev=tokens_by_rev,
            models=models,
            revision_languages=revision_languages,
        )
        if text_mode and "by_task_length" in charts_to_render:
            _print_ascii_bucketed(
                title="# Score by Task Length" if not nolegend else "",
                bucket_scores=bucket_scores,
                bucket_task_counts=bucket_counts,
                models=models,
                width=50,
                model_colors=model_colors_global,
                color_lines=use_color,
            )
        elif "by_task_length" in charts_to_render:
            _plot_by_bucket(
                bucket_scores,
                bucket_counts,
                models=models,
                title=f"Scores by Task Length (Projects: {project_label}, {n_runs} runs)" if not nolegend else "",
                model_colors_global=model_colors_global,
                nolegend=nolegend,
            )

    total_input_tokens = {m: 0 for m in models}
    total_output_tokens = {m: 0 for m in models}
    total_cached_input_tokens = {m: 0 for m in models}
    total_turns = {m: 0 for m in models}
    for comp_rev in valid_revisions:
        for model in models:
            metrics = rev_scores.get(comp_rev, {}).get(model)
            if not metrics:
                continue
            total_input_tokens[model] += int(metrics.get("input_tokens", 0))
            total_output_tokens[model] += int(metrics.get("output_tokens", 0))
            total_cached_input_tokens[model] += int(metrics.get("cached_input_tokens", 0))
            total_turns[model] += int(metrics.get("turns", 0))

    overall_rows: list[dict] = []
    for m in models:
        overall_row = {
            "model": m,
            "n_tasks": n_revisions,
            "task_count": int(n_revisions),
            "pass_count": int(pass_counts.get(m, 0)),
            "build_failures": int(total_build_failures.get(m, 0)),
            "score_percent": float(score_percent.get(m, 0.0)),
            "spend": float(estimated_solve_all_spend_all_models.get(m, float("nan"))),
            "latency_seconds": float(estimated_solve_all_latency_all_models.get(m, float("nan"))),
            "calibration_input_factor": float(calibration_input_factors.get(m, float("nan"))),
            "calibration_output_factor": float(calibration_output_factors.get(m, float("nan"))),
            "calibration_latency_factor": float(calibration_latency_factors.get(m, float("nan"))),
            "calibration_version": CALIBRATION_VERSION,
            "slice_ratio_error": (
                float(per_model_slice_error[m])
                if m in per_model_slice_error and math.isfinite(per_model_slice_error[m])
                else None
            ),
            "total_input_tokens": int(total_input_tokens.get(m, 0)),
            "total_output_tokens": int(total_output_tokens.get(m, 0)),
            "total_cached_input_tokens": int(total_cached_input_tokens.get(m, 0)),
            "total_turns": int(total_turns.get(m, 0)),
            "total_build_failures": int(total_build_failures.get(m, 0)),
        }
        _attach_language_payloads(
            overall_row,
            language_totals=language_totals,
            language_order=language_order,
            model=m,
            task_count_key="task_count",
            include_spend=True,
            include_latency=True,
            include_build_failures=True,
            include_input_tokens=True,
            include_output_tokens=True,
            include_cached_input_tokens=True,
            include_turns=True,
        )
        overall_rows.append(overall_row)
    if export_dir is not None:
        _write_jsonl(export_dir / "overall.jsonl", overall_rows)

    by_tokens_rows: list[dict] = []
    if bucket_scores:
        def _bucket_key(label: str) -> int:
            lp = label.split("-")[0]
            try:
                return int(lp)
            except ValueError:
                return 0

        for bucket in sorted(bucket_scores.keys(), key=_bucket_key):
            tasks_in_bucket = int(bucket_counts.get(bucket, 0))
            for m in models:
                score_sum = float(bucket_scores.get(bucket, {}).get(m, 0.0))
                score_pct = (score_sum / tasks_in_bucket) * 100.0 if tasks_in_bucket else 0.0
                by_tokens_row = {
                    "bucket": bucket,
                    "model": m,
                    "tasks_in_bucket": tasks_in_bucket,
                    "score_percent": score_pct,
                }
                _attach_language_payloads(
                    by_tokens_row,
                    language_totals=bucket_language_totals.get(bucket, {}),
                    language_order=language_order,
                    model=m,
                    task_count_key="tasks_in_bucket",
                )
                by_tokens_rows.append(by_tokens_row)
    if export_dir is not None:
        _write_jsonl(export_dir / "by_tokens.jsonl", by_tokens_rows)

    score_vs_spend_rows: list[dict] = []
    for m in models:
        score_vs_spend_row = {
            "model": m,
            "spend": float(estimated_solve_all_spend_all_models.get(m, float("nan"))),
            "score_percent": float(score_percent.get(m, 0.0)),
            "calibration_version": CALIBRATION_VERSION,
            "slice_ratio_error": (
                float(per_model_slice_error[m])
                if m in per_model_slice_error and math.isfinite(per_model_slice_error[m])
                else None
            ),
        }
        _attach_language_payloads(
            score_vs_spend_row,
            language_totals=language_totals,
            language_order=language_order,
            model=m,
            include_spend=True,
        )
        score_vs_spend_rows.append(score_vs_spend_row)
    if export_dir is not None:
        _write_jsonl(export_dir / "score_vs_spend.jsonl", score_vs_spend_rows)

    score_vs_speed_rows: list[dict] = []
    for m in models:
        score_vs_speed_row = {
            "model": m,
            "latency_seconds": float(estimated_solve_all_latency_all_models.get(m, float("nan"))),
            "score_percent": float(score_percent.get(m, 0.0)),
            "calibration_version": CALIBRATION_VERSION,
            "slice_ratio_error": (
                float(per_model_slice_error[m])
                if m in per_model_slice_error and math.isfinite(per_model_slice_error[m])
                else None
            ),
        }
        _attach_language_payloads(
            score_vs_speed_row,
            language_totals=language_totals,
            language_order=language_order,
            model=m,
            include_latency=True,
        )
        score_vs_speed_rows.append(score_vs_speed_row)
    if export_dir is not None:
        _write_jsonl(export_dir / "score_vs_speed.jsonl", score_vs_speed_rows)

    if not text_mode:
        plt.show()


if __name__ == "__main__":
    main()
