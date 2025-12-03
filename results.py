#!/usr/bin/env python3
"""
Graph benchmark results.

Given a directory that contains JSON files named {model}-{revision}.json,
this script aggregates results for each model where `stopReason == "SUCCESS"`,
computes the score

    score = 1 / log2(...)

and plots the SUM of scores per model.  Each bar is annotated with both the
total score and the total elapsed time (in seconds) accumulated across all
successful revisions.

Usage:
    python results.py /path/to/benchmark/dir
"""

import argparse
import json
import math
import sys
import os
from collections.abc import Callable
from typing import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.ticker as mticker
import litellm
from litellm import completion_cost
from litellm.utils import CostPerToken
litellm.suppress_debug_info = True

# Load pretty names for models from model_metadata.json (if available).
# This avoids hardcoding display names and keeps chart labels consistent.
_MODEL_METADATA_PATH = Path(__file__).parent / "model_metadata.json"
try:
    _MODEL_METADATA = json.loads(_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    _MODEL_ALIASES: dict = {}
    if isinstance(_MODEL_METADATA, dict):
        _MODEL_ALIASES = _MODEL_METADATA.get("model_aliases", {}) or {}
except Exception:
    _MODEL_ALIASES = {}

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

    'gpt5-high': 'gpt5',
    'gpt5-nothink': 'gpt5',

    'gpt5-mini-low': 'gpt5-mini',
    'gpt5-mini-high': 'gpt5-mini',

    'gpt5.1-high': 'gpt5.1',
    'gpt5.1-nothink': 'gpt5.1',

    'gpt5-nano-high': 'gpt5-nano',

    "opus4.1-high": 'opus4.1',
    "opus4.1-nothink": 'opus4.1',

    "dsv3.2-exp": "dsv3.2",
    "dsr3.2-exp": "dsr3.2",
}

# Custom per-token pricing for models not recognised by LiteLLM
CUSTOM_MODEL_PRICING: dict[str, CostPerToken] = {
    "sonnet4.5":    {"input_cost_per_token": 3e-06,  "output_cost_per_token": 1.5e-05},
    "sonnet4.5-high": {"input_cost_per_token": 3e-06,  "output_cost_per_token": 1.5e-05},
    "haiku4.5":    {"input_cost_per_token": 1e-06,  "output_cost_per_token": 5e-06},
    "haiku4.5-low": {"input_cost_per_token": 1e-06,  "output_cost_per_token": 5e-06},
    "haiku4.5-nothink": {"input_cost_per_token": 1e-06,  "output_cost_per_token": 5e-06},
    "dsv3.2":       {"input_cost_per_token": 2.8e-07,  "output_cost_per_token": 4.2e-07},
    "dsr3.2":       {"input_cost_per_token": 2.8e-07,  "output_cost_per_token": 4.2e-07},
    "m2":           {"input_cost_per_token": 3e-07,  "output_cost_per_token": 1.2e-06},
    "k2":           {"input_cost_per_token": 1e-06,  "output_cost_per_token": 3e-06},
    "k2-thinking":  {"input_cost_per_token": 1e-06,  "output_cost_per_token": 2.5e-06},
    "gpt-oss-20b":  {"input_cost_per_token": 5e-08,"output_cost_per_token": 2e-07},
    "gpt-oss-120b": {"input_cost_per_token": 1.5e-07,"output_cost_per_token": 6e-07},
    "grok-3":       {"input_cost_per_token": 3e-06,  "output_cost_per_token": 1.5e-6},
    "grok-3-mini":  {"input_cost_per_token": 3e-07,  "output_cost_per_token": 5e-7},
    "grok-3-mini-high":  {"input_cost_per_token": 3e-07,  "output_cost_per_token": 5e-7},
    "gcf1":         {"input_cost_per_token": 2e-07,"output_cost_per_token": 1.5e-06},
    "grok4-fast":   {"input_cost_per_token": 2e-07,"output_cost_per_token": 5e-07},
    "grok4.1-fast": {"input_cost_per_token": 2e-07,"output_cost_per_token": 5e-07},
    "grok4":        {"input_cost_per_token": 3e-06,"output_cost_per_token": 1.5e-05},
    "opus4.1":      {"input_cost_per_token": 1.5e-05,"output_cost_per_token": 7.5e-5},
    "opus4.5":      {"input_cost_per_token": 5e-06,"output_cost_per_token": 2.5e-5},
    "q3c":          {"input_cost_per_token": 1.5e-06,  "output_cost_per_token": 7.5e-06},
    "q3c-fp8":      {"input_cost_per_token": 4e-07,  "output_cost_per_token": 1.6e-06},
    "q3c-30b":      {"input_cost_per_token": 1e-07,  "output_cost_per_token": 3e-07},
    "q3next":       {"input_cost_per_token": 1.5e-07,  "output_cost_per_token": 1.5e-06},
    "qwen3-max":    {"input_cost_per_token": 1.2e-06,  "output_cost_per_token": 6e-06},
    "glm4.5":       {"input_cost_per_token": 6e-07,  "output_cost_per_token": 2.2e-06},
    "glm4.6":       {"input_cost_per_token": 6e-07,  "output_cost_per_token": 2.2e-06},
    "glm4.6-fp8":   {"input_cost_per_token": 4.5e-07,  "output_cost_per_token": 2.0e-06},
    "glm4.5-air":   {"input_cost_per_token": 2e-07,  "output_cost_per_token": 1.1e-06},
    "r1":           {"input_cost_per_token": 5.5e-07,  "output_cost_per_token": 2.19e-06},
    "ds-r1.1":      {"input_cost_per_token": 5.5e-07,  "output_cost_per_token": 2.19e-06},
    "gpt5":         {"input_cost_per_token": 1.25e-06,"output_cost_per_token": 1e-05},
    "gpt5.1":       {"input_cost_per_token": 1.25e-06,"output_cost_per_token": 1e-05},
    "gpt5-codex":   {"input_cost_per_token": 1.25e-06,"output_cost_per_token": 1e-05},
    "gpt5-mini":    {"input_cost_per_token": 2.5e-07,"output_cost_per_token": 2e-06},
    "gpt5-nano":    {"input_cost_per_token": 5e-08,"output_cost_per_token": 4e-07},
    "gp3":          {"input_cost_per_token": 2e-06,  "output_cost_per_token": 1.2e-05},
    "dsv3.2":       {"input_cost_per_token": 2.8e-07,  "output_cost_per_token": 4.2e-07},
}

MODEL_PRICING_OVER_200k: dict[str, CostPerToken] = {
    "gp3": {"input_cost_per_token": 4.0e-06,  "output_cost_per_token": 1.8e-05},
}

# Palette used for model bars. Cycles these colors as necessary.
# Models:
#   "#EB3F33", // Primary red
#   "#3B82F6", // Blue
#   "#10B981", // Green
#   "#F59E0B", // Amber
#   "#8B5CF6", // Purple
#   "#EC4899", // Pink
#   "#14B8A6", // Teal
#   "#6B7280", // Gray
#   "#F87171", // Light red
#   "#60A5FA", // Light blue
#   "#34D399", // Light green
#   "#FBBF24", // Light amber
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


def _get_distinct_colors(n: int) -> list[str]:
    """
    Return `n` colours from the model-bar palette, cycling as required.

    The palette is intentionally explicit and will be repeated to supply as
    many distinct colours as requested.
    """
    repeats = (n + len(_MODEL_BAR_PALETTE) - 1) // len(_MODEL_BAR_PALETTE)
    return (_MODEL_BAR_PALETTE * repeats)[:n]


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


def _parse_args() -> tuple[list[tuple[str, Path]], list[str] | None, set[str] | None, list[str], int | None, bool, set[str]]:
    """
    Parse command-line arguments.

    Returns
    -------
    tuple
        (run_info, models_order, exclude_models, projects, tasksize, text, charts) where:
          - run_info is a list of (project, Path) tuples, one per run directory
          - models_order is a list of model names supplied via ``--models`` (in order, with "_" placeholders) or ``None``
          - exclude_models is a set of model names supplied via ``--exclude`` or ``None``
          - projects is the list of project name prefixes
          - tasksize is an integer maximum token count (or None if not provided)
          - text is a boolean; when True, prints ASCII charts to stdout instead of using matplotlib
          - charts is a set of chart names to render (e.g. {'mainscore', 'score_v_latency'})
    """
    parser = argparse.ArgumentParser(description="Graph benchmark results.")
    parser.add_argument(
        "base_directory",
        type=Path,
        help="Base directory containing project run directories (e.g. coderesults/).",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Comma-separated list of project names, prefixes for run directories (e.g. brokk,gizmo).",
    )
    parser.add_argument(
        "--runs",
        metavar="LIST",
        help="Comma-separated list of run numbers to include (e.g. 1,2,4). "
        "If omitted, all numeric-suffixed runs for the given projects are processed.",
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
        "Available: mainscore, per_task_best, per_task_avg, llm_runtime, score_v_spend, score_v_latency, score_speed_price, by_task_length_best, by_task_length_avg. "
        "Default: mainscore,score_v_latency,score_v_spend,score_speed_price",
    )
    args = parser.parse_args()

    if not args.base_directory.is_dir():
        parser.error(f"{args.base_directory} is not a directory")

    projects = [p.strip() for p in str(args.project).split(",") if p.strip()]
    if not projects:
        parser.error("--project must be a comma-separated list of one or more project names.")

    charts = {c.strip() for c in str(args.charts).split(",") if c.strip()}
    if not charts:
        parser.error("--charts provided but no valid chart names parsed")
    valid_charts = {
        "mainscore",
        "per_task_best",
        "per_task_avg",
        "llm_runtime",
        "score_v_spend",
        "score_v_latency",
        "score_speed_price",
        "by_task_length_best",
        "by_task_length_avg",
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
                if d.is_dir() and d.name.startswith(project_name):
                    run_suffix = d.name[len(project_name) :]
                    if run_suffix.isdigit():
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

    tasksize: int | None = None
    if args.tasksize is not None:
        if args.tasksize < 0:
            parser.error("--tasksize must be non-negative")
        tasksize = int(args.tasksize)

    return run_info, models_order, exclude_models, projects, tasksize, bool(args.text), charts


def _model_from_filename(path: Path) -> str:
    """
    Extract model name from a filename of the form <model>-<revision>.json.

    Handles models whose names themselves contain dashes by splitting
    on the LAST dash.
    """
    stem = path.stem  # filename without extension
    model, sep, _revision = stem.rpartition("-")
    return model if sep else stem  # if no dash found, whole stem is the model


def _aggregate(
    directory: Path,
    include_models: set[str] | None = None,
    exclude_models: set[str] | None = None,
    tasksize: int | None = None,
):
    """
    Read all JSON files inside `directory` and aggregate information per model.

    When `tasksize` is provided (an integer), only revisions whose corresponding
    codetasks/<revision>.json contains a valid integer "tokens" value <= tasksize
    are considered. Revisions missing token metadata or whose tokens exceed the
    limit will be skipped (with a warning printed to stderr).

    Returns a tuple:
        scores: dict[model, float]   summed score across successful revisions
        elapsed: dict[model, int]    summed elapsedMillis across successful revisions
        costs: dict[model, float]    summed cost across successful revisions
        n_revisions: int             number of distinct revisions that must be present
        rev_scores: dict[revision, dict[model, dict]]  per-revision details used
            by the task-level plot.  Each inner dict contains:
                {
                    "score":   float,
                    "build":   int,
                    "parse":   int,
                    "edit":    int,
                    "elapsed": int,
                    "cost":    float,
                }

    The function raises a ValueError if **any** model does not have the exact same
    set of revision files as every other model in the directory (after applying filters).
    """
    scores: dict[str, float] = {}
    elapsed: dict[str, int] = {}
    costs: dict[str, float] = {}
    revisions_by_model: dict[str, set[str]] = {}
    rev_scores: dict[str, dict[str, dict]] = {}

    found_models: set[str] = set()  # Models that passed all filters and were processed

    # Try to discover a nearby codetasks directory for this run directory
    codetasks_dir = next(
        (c for c in (directory.parent / "codetasks", directory.parent.parent / "codetasks") if c.is_dir()),
        None,
    )

    for json_file in directory.glob("*.json"):
        model = _model_from_filename(json_file)

        # Apply include filter (whitelist)
        if include_models is not None and model not in include_models:
            continue

        # Apply exclude filter (blacklist)
        if exclude_models is not None and model in exclude_models:
            continue

        # This model passed all filters
        found_models.add(model)
        _model, _sep, revision = json_file.stem.rpartition("-")

        # If tasksize filtering is enabled, consult the codetasks metadata
        if tasksize is not None:
            if codetasks_dir is None:
                print(f"Warning: Could not locate codetasks/ for run directory {directory}; skipping revision '{revision}' due to --tasksize.", file=sys.stderr)
                continue
            task_file = codetasks_dir / f"{revision}.json"
            if not task_file.is_file():
                print(f"Warning: Missing codetasks file {task_file}; skipping revision '{revision}' due to --tasksize.", file=sys.stderr)
                continue
            try:
                task_data = json.loads(task_file.read_text())
                tokens_val = task_data.get("tokens")
                if not isinstance(tokens_val, int) or tokens_val < 0:
                    print(f"Warning: Invalid tokens value in {task_file}; skipping revision '{revision}' due to --tasksize.", file=sys.stderr)
                    continue
                if tokens_val > tasksize:
                    # Skip tasks larger than the requested task size
                    continue
            except Exception:
                print(f"Warning: Failed to read tokens from {task_file}; skipping revision '{revision}' due to --tasksize.", file=sys.stderr)
                continue

        # Track revisions encountered for each model
        revisions_by_model.setdefault(model, set()).add(revision)

        try:
            data = json.loads(json_file.read_text())

            # Compute cost for this revision irrespective of success/failure
            cost = 0.0
            prompt_tokens = data.get("inputTokens", 0)
            completion_tokens = data.get("outputTokens", 0)
            cache_read_tokens = data.get("cachedInputTokens", 0)
            
            response_for_cost = {
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cache_read_input_tokens": cache_read_tokens,
                }
            }

            # Determine which pricing to use based on prompt token count
            mapped_model = MODEL_MAPPING.get(model, model)
            pricing_to_use = None
            
            # Check if we should use pricing for >200k tokens
            if prompt_tokens > 200000 and mapped_model in MODEL_PRICING_OVER_200k:
                pricing_to_use = MODEL_PRICING_OVER_200k[mapped_model]
            # Otherwise prefer custom pricing if defined for this model
            elif mapped_model in CUSTOM_MODEL_PRICING:
                pricing_to_use = CUSTOM_MODEL_PRICING[mapped_model]
            
            if pricing_to_use is not None:
                cost = completion_cost(
                    completion_response=response_for_cost,
                    model=mapped_model,  # model may be unknown to LiteLLM
                    custom_cost_per_token=pricing_to_use,
                    custom_pricing=True,
                )
            else:
                cost = completion_cost(
                    completion_response=response_for_cost,
                    model=mapped_model,
                )
            # Only warn if cost is 0 for a task that had actual edits
            if not cost and data.get("editBlocksTotal", 0) > 0:
                print(f"Warning: Cost is 0 for model '{model}' revision '{revision}'. Check pricing configuration.", file=sys.stderr)
        except json.JSONDecodeError as exc:
            print(f"Skipping {json_file}: invalid JSON ({exc})", file=sys.stderr)
            continue

        if data.get("stopReason") == "SUCCESS":
            build_failures = data.get("buildFailures", 0)
            parse_failures = data.get("parseRetries", 0)
            apply_failures = data.get("applyRetries", 0)
            reflections = data.get("reflections", 0)
            retries = build_failures + parse_failures + apply_failures + reflections
            penalty = build_failures + 2
            score = 1.0 / math.log2(penalty)
            time_ms = data.get("elapsedMillis", 0)
            llm_ms = data.get("llmMillis", 0)

            # cost already computed above

            rev_metrics = {
                "score": score,
                "retries": retries,
                "build": build_failures,
                "parse": parse_failures,
                "apply": apply_failures,
                "elapsed": time_ms,
                "llm": llm_ms,
                "cost": cost,
            }
        else:
            # Failed runs score 0
            score = 0.0
            time_ms = 0
            # cost already computed above for failed runs too
            rev_metrics = {
                "score": 0.0,
                "retries": 0,
                "build": 0,
                "parse": 0,
                "apply": 0,
                "elapsed": 0,
                "llm": data.get("llmMillis", 0),
                "cost": cost,
            }

        # aggregate per-model sums
        scores[model] = scores.get(model, 0.0) + score
        elapsed[model] = elapsed.get(model, 0) + time_ms
        costs[model] = costs.get(model, 0.0) + cost
        rev_scores.setdefault(revision, {})[model] = rev_metrics

    if not revisions_by_model:
        raise ValueError(
            "No benchmark files matched the given criteria "
            f"in {directory}"
        )

    # Build the union of revisions across the models present (after filters)
    present_models = set(revisions_by_model.keys())
    if include_models is not None:
        models_to_use = sorted((include_models - (exclude_models or set())) & present_models)
    else:
        models_to_use = sorted(present_models)

    all_revisions: set[str] = set()
    for m in models_to_use:
        all_revisions |= revisions_by_model[m]

    n_revisions = len(all_revisions)
    return scores, elapsed, costs, n_revisions, rev_scores


def _compute_stats(
    merged_rev_scores: dict[str, dict[str, list[dict]]],
) -> tuple[dict, dict]:
    """
    Compute average and best scores from merged revision scores over multiple runs.

    Returns a tuple (avg_rev_scores, best_rev_scores).
    """
    avg_rev_scores: dict[str, dict[str, dict]] = {}
    best_rev_scores: dict[str, dict[str, dict]] = {}

    for rev, model_data in merged_rev_scores.items():
        avg_rev_scores[rev] = {}
        best_rev_scores[rev] = {}
        for model, metrics_list in model_data.items():
            if not metrics_list:
                continue

            # Average scores
            num_runs = len(metrics_list)
            avg_score = sum(m["score"] for m in metrics_list) / num_runs
            avg_cost = sum(m["cost"] for m in metrics_list) / num_runs
            avg_retries = int(round(sum(m["retries"] for m in metrics_list) / num_runs))
            avg_build   = int(round(sum(m["build"]   for m in metrics_list) / num_runs))
            avg_parse   = int(round(sum(m["parse"]   for m in metrics_list) / num_runs))
            avg_apply   = int(round(sum(m["apply"]   for m in metrics_list) / num_runs))
            avg_elapsed = sum(m["elapsed"] for m in metrics_list) / num_runs
            avg_llm = sum(m["llm"] for m in metrics_list) / num_runs

            avg_rev_scores[rev][model] = {
                "score": avg_score,
                "cost": avg_cost,
                "retries": avg_retries,
                "build": avg_build,
                "parse": avg_parse,
                "apply": avg_apply,
                "elapsed": avg_elapsed,
                "llm": avg_llm,
            }

            # Best score
            best_run_metrics = max(metrics_list, key=lambda m: m["score"])
            best_rev_scores[rev][model] = best_run_metrics

    return avg_rev_scores, best_rev_scores


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


def _desaturate_hsv(hex_color: str, saturation_factor: float) -> str:
    """
    Reduce saturation of a hex color by multiplying the S channel in HSV space.
    
    Args:
        hex_color: Hex color string (e.g., "#FF0000")
        saturation_factor: Multiplicative factor for saturation (0.0 to 1.0)
    
    Returns:
        Desaturated hex color string
    """
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    # Convert RGB to HSV
    h, s, v = mcolors.rgb_to_hsv([r, g, b])
    
    # Reduce saturation
    s = s * saturation_factor
    
    # Convert back to RGB
    rgb = mcolors.hsv_to_rgb([h, s, v])
    r, g, b = rgb[0] * 255, rgb[1] * 255, rgb[2] * 255
    
    # Convert to hex
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _plot(
    sum_avg_scores: dict[str, float],
    sum_best_scores: dict[str, float],
    sum_avg_costs: dict[str, float],
    sum_best_costs: dict[str, float],
    models: list[str],
    n_revisions: int,
    project_name: str,
    n_runs: int,
    base_dir_name: str,
    model_colors_global: dict[str, str],
) -> None:
    if not models:
        print("No successful benchmark results found.", file=sys.stderr)
        sys.exit(1)

    # Normalize scores to percentages
    avg_scores_vals = [
        (sum_avg_scores[m] / n_revisions) * 100 for m in models
    ]
    best_scores_vals = [
        (sum_best_scores[m] / n_revisions) * 100 for m in models
    ]

    # Sort models by average score descending
    sorted_indices = sorted(range(len(models)), key=lambda i: avg_scores_vals[i], reverse=True)
    models_sorted = [models[i] for i in sorted_indices]
    avg_scores_vals_sorted = [avg_scores_vals[i] for i in sorted_indices]
    best_scores_vals_sorted = [best_scores_vals[i] for i in sorted_indices]

    # Get per-model colors from global mapping
    model_colors = [model_colors_global[m] for m in models_sorted]
    
    # Desaturate best colors by 30% (multiply saturation by 0.7)
    best_colors = [_desaturate_hsv(c, 0.7) for c in model_colors]

    y = np.arange(len(models_sorted))
    height = 0.35
    fig, ax = plt.subplots(figsize=(10, max(6, 0.5 * len(models_sorted))))
    fig.canvas.mpl_connect('key_press_event', _on_key_press)

    # Draw horizontal bars (avg on top, best on bottom)
    # Because the y-axis is inverted, smaller y-values appear higher on the chart.
    rects1 = ax.barh(
        y - height / 2, avg_scores_vals_sorted, height, color=model_colors
    )
    rects2 = ax.barh(
        y + height / 2, best_scores_vals_sorted, height, color=best_colors
    )

    ax.set_xlabel("Score (%)")
    ax.set_title("Scores by model")
    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_name(m) for m in models_sorted])
    ax.invert_yaxis()  # Highest score at top
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))

    # Annotate bars
    for i, model in enumerate(models_sorted):
        # Avg bar
        w_avg = rects1[i].get_width()
        ax.text(
            w_avg,
            rects1[i].get_y() + rects1[i].get_height() / 2,
            f" {w_avg:.2f}% (Average)",
            ha="left",
            va="center",
            fontsize=9,
        )

        # Best bar
        w_best = rects2[i].get_width()
        ax.text(
            w_best,
            rects2[i].get_y() + rects2[i].get_height() / 2,
            f" {w_best:.2f}% (Best)",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()


def _plot_by_revision(
    rev_scores: dict[str, dict[str, dict]], models: list[str], title: str, model_colors_global: dict[str, str],
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
            # Special handling for 0-score runs to avoid overlapping text
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
    ax.set_title(title)
    ax.set_xlim(0, 1)

    legend_handles = [
        plt.matplotlib.patches.Patch(color=cmap(i), label=_pretty_name(model))
        for i, model in enumerate(models)
    ]
    ax.legend(handles=legend_handles, title="Models", loc="upper right", fontsize=8)

    plt.tight_layout()


def _plot_llm_by_revision(
    merged_rev_scores: dict[str, dict[str, list[dict]]],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str],
) -> None:
    """
    Draw a horizontal grouped-bar chart with one group per revision (task) and one
    bar per model inside that group, where the x-axis is normalized LLM runtime.

    For each task (revision) and model, we SUM llmMillis across all runs
    (including successes and failures). We normalize by dividing by that model's
    fastest successful single-run llmMillis observed across any task.
    """
    if not merged_rev_scores:
        return

    # Build denominator: per-model fastest successful single-run llmMillis across any task
    min_success_llm_ms: dict[str, float] = {}
    for model in models:
        llms: list[float] = []
        for _rev, model_data in merged_rev_scores.items():
            metrics_list = model_data.get(model, [])
            for m in metrics_list:
                if m.get("score", 0.0) > 0 and m.get("llm", 0) > 0:
                    llms.append(float(m["llm"]))
        if llms:
            min_success_llm_ms[model] = min(llms)

    active_models = [m for m in models if m in min_success_llm_ms]
    if not active_models:
        # Nothing to plot if no model has any successful LLM runtimes
        return

    # Sort revisions from slowest to fastest using the median of normalized LLM runtime ratios across models.
    # For each task and model, we use:
    #   ratio = (sum of llmMillis across all runs for that task and model) / (model's fastest successful single-run llmMillis across any task)
    rev_median_ratio: dict[str, float] = {}
    for comp_rev, model_data in merged_rev_scores.items():
        ratios: list[float] = []
        for m in active_models:
            metrics_list = model_data.get(m, [])
            total_llm_ms = float(sum(mm.get("llm", 0.0) for mm in metrics_list))
            denom = float(min_success_llm_ms[m])
            ratio = (total_llm_ms / denom) if denom > 0 else 0.0
            ratios.append(ratio)
        rev_median_ratio[comp_rev] = float(np.median(ratios)) if ratios else 0.0

    sorted_revs = sorted(
        merged_rev_scores.items(),
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
            metrics_list = model_data.get(model, [])
            # Sum llmMillis across all runs (success + failure) for this task
            total_llm_ms = float(sum(m.get("llm", 0.0) for m in metrics_list))

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
    ax.set_title(title)
    ax.set_xlim(0, max_ratio * 1.05 if max_ratio > 0 else 1.0)

    legend_handles = [
        plt.matplotlib.patches.Patch(color=cmap(i), label=_pretty_name(model))
        for i, model in enumerate(active_models)
    ]
    ax.legend(handles=legend_handles, title="Models", loc="upper right", fontsize=8)

    plt.tight_layout()


def _plot_scatter(
    x_data: dict[str, float],
    y_data: dict[str, float],
    models: list[str],
    title: str,
    x_label: str,
    y_label: str,
    x_log_base: int | None = None,
    model_colors_global: dict[str, str] | None = None,
) -> None:
    """
    Draw a scatter plot with labels for each point.
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

        ax.scatter(x_val, y_val, color=cmap(i))
        ax.text(x_val, y_val, f"  {_pretty_name(model)}", fontsize=9, va="center")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_ylim(bottom=0)
    if x_log_base:
        ax.set_xscale("log", base=x_log_base)
    else:
        ax.set_xlim(left=0)

    legend_handles = [
        plt.matplotlib.patches.Patch(color=cmap(i), label=_pretty_name(model))
        for i, model in enumerate(models)
    ]
    ax.legend(handles=legend_handles, title="Models", loc="best", fontsize=8)

    plt.tight_layout()


def _compute_success_only_spend(
    merged_rev_scores: dict[str, dict[str, list[dict]]],
    models: list[str],
) -> dict[str, float]:
    """
    Compute success-only spend for each model.
    
    Returns success_spend[model] = sum of costs for only successful runs.
    """
    success_spend = {m: 0.0 for m in models}
    
    for _rev, model_data in merged_rev_scores.items():
        for model in models:
            metrics_list = model_data.get(model, [])
            for metric in metrics_list:
                if metric.get("score", 0.0) > 0:
                    success_spend[model] += metric.get("cost", 0.0)
    
    return success_spend


def _compute_success_only_latency(
    merged_rev_scores: dict[str, dict[str, list[dict]]],
    models: list[str],
) -> dict[str, float]:
    """
    Compute success-only LLM latency for each model (in seconds).
    
    Returns success_latency_s[model] = sum of llmMillis for only successful runs (in seconds).
    """
    success_latency = {m: 0.0 for m in models}
    
    for _rev, model_data in merged_rev_scores.items():
        for model in models:
            metrics_list = model_data.get(model, [])
            for metric in metrics_list:
                if metric.get("score", 0.0) > 0:
                    llm_ms = metric.get("llm", 0.0)
                    success_latency[model] += llm_ms / 1000.0
    
    return success_latency


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
    success_latency: dict[str, float],
    y_data: dict[str, float],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str] | None = None,
) -> None:
    """
    Scatter plot displaying latency for successful tasks only.
    """
    _plot_success_scatter(
        x_data=success_latency,
        y_data=y_data,
        models=models,
        title=title + " (Successful Tasks)",
        x_label="LLM Latency (s)",
        x_formatter=_format_seconds,
        x_axis_formatter=mticker.FuncFormatter(lambda x, _p: f"{x:,.0f}"),
        model_colors_global=model_colors_global,
    )


def _plot_spend_scatter(
    success_spend: dict[str, float],
    y_data: dict[str, float],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str] | None = None,
) -> None:
    """
    Scatter plot displaying spend for successful tasks only.
    """
    _plot_success_scatter(
        x_data=success_spend,
        y_data=y_data,
        models=models,
        title=title + " (Successful Tasks)",
        x_label="Spend ($)",
        x_formatter=lambda cost: f"${cost:.2f}",
        model_colors_global=model_colors_global,
    )


# ---------- Task-length bucketing utilities ----------

_BUCKET_RANGES: list[tuple[int, int | None]] = [
    (0, 256),
    (257, 512),
    (513, 1024),
    (1025, 2048),
    (2049, 4096),
    (4097, None),  # open-ended upper range
]


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


def _load_revision_tokens(
    revisions: Iterable[str],
    codetasks_dir: Path,
) -> dict[str, int]:
    """
    Load the *tokens* count for each revision from ``codetasks/<revision>.json``.
    """
    tokens_by_rev: dict[str, int] = {}
    for rev in revisions:
        task_file = codetasks_dir / f"{rev}.json"
        if not task_file.is_file():
            continue
        try:
            data = json.loads(task_file.read_text())
            tokens_val = data.get("tokens")
            if isinstance(tokens_val, int) and tokens_val >= 0:
                tokens_by_rev[rev] = tokens_val
        except Exception:
            continue
    return tokens_by_rev


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


def _plot_radar(
    normalized_score: dict[str, float],
    normalized_speed: dict[str, float],
    normalized_price: dict[str, float],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str],
) -> None:
    """
    Draw a radar plot (spider chart) with three axes (Score, Speed, Price),
    each normalized to [0, 100] where 100% = best on each dimension.
    
    Each model is represented as a filled polygon connecting its three normalized values.
    """
    if not models:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.canvas.mpl_connect("key_press_event", _on_key_press)
    
    # Three axes for the radar
    axes_names = ["Score", "Speed", "Price"]
    num_axes = len(axes_names)
    
    # Angle for each axis (in radians)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    # Close the plot by repeating the first angle
    angles += angles[:1]
    
    n_models = len(models)
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
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()


def _plot_by_bucket(
    bucket_scores: dict[str, dict[str, float]],
    bucket_task_counts: dict[str, int],
    models: list[str],
    title: str,
    model_colors_global: dict[str, str],
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
    ax.set_title(title)
    ax.set_xlim(0, 100) # Set x-limit to 100%

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
    Heuristic: treat Jupyter inline backends as display; Windows as having a display;
    and on Unix require DISPLAY to be set.
    """
    try:
        backend = (plt.get_backend() or "").lower()
    except Exception:
        backend = ""
    if "inline" in backend:
        return True
    if sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY"))


def _bar_str(value: float, max_value: float, width: int, formatter: Callable[[float], str]) -> str:
    """
    Build a horizontal ASCII bar for a given value in [0, max_value].
    """
    if max_value <= 0:
        filled = 0
    else:
        filled = int(round((value / max_value) * width))
    bar = "#" * max(0, min(filled, width))
    return f"{bar} {formatter(value)}"


def _print_ascii_dual_series(
    title: str,
    models: list[str],
    series1_name: str,
    series1_vals: dict[str, float],
    series2_name: str,
    series2_vals: dict[str, float],
    width: int,
    formatter: Callable[[float], str],
    normalize_to_100: bool = False,
) -> None:
    """
    Print two bars per model (e.g., Avg vs Best or All vs Success).

    If `normalize_to_100` is True the value 100.0 is treated as the full bar
    length (useful for percentage-based series). Otherwise the maximum value
    across both series is used as the full bar length.
    """
    print('# ' + title)
    max_label = max((len(_pretty_name(m)) for m in models), default=0)
    max_name = max(len(series1_name), len(series2_name))
    if normalize_to_100:
        max_value = 100.0
    else:
        max_value = max(
            [series1_vals.get(m, 0.0) for m in models]
            + [series2_vals.get(m, 0.0) for m in models]
            + [0.0]
        )
    for m in models:
        display = _pretty_name(m)
        v1 = float(series1_vals.get(m, 0.0))
        v2 = float(series2_vals.get(m, 0.0))
        bar1 = _bar_str(v1, max_value, width, formatter)
        bar2 = _bar_str(v2, max_value, width, formatter)
        print(f"{display:<{max_label}} | {series1_name:<{max_name}} {bar1}")
        print(f"{'':<{max_label}} | {series2_name:<{max_name}} {bar2}")
    print("")


def _print_ascii_single_series(
    title: str,
    labels: list[str],
    values: dict[str, float],
    width: int,
    formatter: Callable[[float], str],
) -> None:
    """
    Print one bar per label.
    """
    print('# ' + title)
    max_label = max((len(_pretty_name(label)) for label in labels), default=0)
    max_value = max([values.get(label, 0.0) for label in labels] + [0.0])
    for label in labels:
        display = _pretty_name(label)
        v = float(values.get(label, 0.0))
        bar = _bar_str(v, max_value, width, formatter)
        print(f"{display:<{max_label}} | {bar}")
    print("")


def _print_ascii_bucketed(
    title: str,
    bucket_scores: dict[str, dict[str, float]],
    bucket_task_counts: dict[str, int],
    models: list[str],
    width: int,
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
            display = _pretty_name(m)
            v = vals.get(m, 0.0)
            # Normalize percentages to the 0-100 range so that 100% fills the bar.
            bar = _bar_str(v, 100.0, width, lambda x: f"{x:.2f}%")
            print(f"  {display:<{max_label}} | {bar}")
    print("")


def main() -> None:
    run_info, models_order_from_cli, exclude_models_filter, projects, tasksize, force_text, charts_to_render = _parse_args()
    text_mode = bool(force_text) or not _has_display()
    
    # Convert CLI models order to a set for filtering purposes
    models_filter: set[str] | None = None
    if models_order_from_cli is not None:
        models_filter = {m for m in models_order_from_cli if m != "_"}

    run_results: list[tuple[str, Path, tuple]] = []
    for project_name, run_dir in run_info:
        try:
            run_results.append((
                project_name,
                run_dir,
                _aggregate(run_dir, models_filter, exclude_models_filter, tasksize)
            ))
        except ValueError as e:
            print(
                f"Skipping run {run_dir.name} due to error: {e}", file=sys.stderr
            )
            continue

    if not run_results:
        print("No valid run data found.", file=sys.stderr)
        sys.exit(1)

    # Build per-project run mapping
    project_runs: dict[str, list[tuple[Path, dict[str, dict[str, dict]]]]] = {}
    for project_name, run_dir, res in run_results:
        rev_scores = res[4]
        project_runs.setdefault(project_name, []).append((run_dir, rev_scores))

    # Determine which models we care about (after exclusions)
    effective_models: set[str] | None = None
    if models_filter is not None:
        effective_models = models_filter - (exclude_models_filter or set())

    # Filter out runs that are missing any task for any requested model
    complete_run_results: list[tuple[str, Path, tuple]] = []
    for project_name, runs in project_runs.items():
        # Determine models to check for this project
        present_models_in_project: set[str] = set()
        for _rd, rev_scores in runs:
            for _rev, model_data in rev_scores.items():
                present_models_in_project.update(model_data.keys())

        if effective_models is not None:
            models_to_check = set(effective_models)
        else:
            models_to_check = present_models_in_project
        # Apply exclusions even when no explicit inclusion list is given
        if exclude_models_filter is not None:
            models_to_check -= exclude_models_filter

        # Union of tasks across all runs for the models we care about
        union_revs: set[str] = set()
        for _rd, rev_scores in runs:
            for rev, model_data in rev_scores.items():
                if any(m in models_to_check for m in model_data.keys()):
                    union_revs.add(rev)

        # Keep only runs that have every task for every requested model
        for rd, rev_scores in runs:
            # Build missing revisions per model for this run
            missing_by_model: dict[str, list[str]] = {}
            for m in models_to_check:
                missing_revs = [rev for rev in union_revs if m not in rev_scores.get(rev, {})]
                if missing_revs:
                    missing_by_model[m] = sorted(missing_revs)

            if missing_by_model:
                print(f"Skipping run {rd.name} due to missing tasks:", file=sys.stderr)
                for m, misses in sorted(missing_by_model.items()):
                    print(f"  {m}: [{', '.join(misses)}]", file=sys.stderr)
            else:
                # Find the full 'res' tuple corresponding to this run_dir
                res = next(r for (p, rdir, r) in run_results if p == project_name and rdir == rd)
                complete_run_results.append((project_name, rd, res))

    if not complete_run_results:
        print("No valid run data found after enforcing task completeness across runs.", file=sys.stderr)
        sys.exit(1)

    # Determine export directory under the base directory of the remaining runs
    export_dir = complete_run_results[0][1].parent / "jsonexport"

    # Collect per-run rev_scores along with their project (filtered)
    runs_rev_scores: list[tuple[str, dict[str, dict[str, dict]]]] = [
        (project, res[4]) for (project, _rd, res) in complete_run_results
    ]

    all_models: set[str] = set()
    all_revisions: set[str] = set()  # composite "<project>:<revision>"
    for project_name, rev_scores in runs_rev_scores:
        for rev, model_data in rev_scores.items():
            all_revisions.add(f"{project_name}:{rev}")
            all_models.update(model_data.keys())

    # Determine final model order and build color mapping
    if models_order_from_cli is not None:
        # Use CLI order, filtering out "_" placeholders
        models = [m for m in models_order_from_cli if m != "_" and m in all_models]
        # Append any models not in CLI list (in sorted order for determinism)
        remaining = sorted(all_models - set(models))
        models.extend(remaining)
    else:
        # No CLI order specified; use sorted discovery order
        models = sorted(list(all_models))
    
    n_revisions = len(all_revisions)
    
    # Build global color mapping: model -> color from palette
    model_colors_global = {}
    color_palette = _get_distinct_colors(len(models))
    for i, model in enumerate(models):
        model_colors_global[model] = color_palette[i]

    # Merge results across runs without padding zeros from other projects.
    # For each composite revision "<project>:<revision>", aggregate the list of metrics per model
    # from only those runs belonging to that project.
    merged_rev_scores: dict[str, dict[str, list[dict]]] = {
        rev: {model: [] for model in models} for rev in all_revisions
    }
    for project_name, rev_scores in runs_rev_scores:
        for rev, model_data in rev_scores.items():
            comp_rev = f"{project_name}:{rev}"
            for model, metrics in model_data.items():
                merged_rev_scores[comp_rev][model].append(metrics)

    avg_rev_scores, best_rev_scores = _compute_stats(merged_rev_scores)

    # Calculate summed scores for aggregate plot
    sum_avg_scores = {m: 0.0 for m in models}
    sum_best_scores = {m: 0.0 for m in models}
    sum_avg_costs = {m: 0.0 for m in models}
    sum_best_costs = {m: 0.0 for m in models}
    sum_avg_elapsed = {m: 0 for m in models}
    sum_best_elapsed = {m: 0 for m in models}
    sum_avg_llm = {m: 0 for m in models}
    sum_best_llm = {m: 0 for m in models}

    for rev_data in avg_rev_scores.values():
        for model, metrics in rev_data.items():
            sum_avg_scores[model] += metrics["score"]
            sum_avg_costs[model] += metrics["cost"]
            sum_avg_elapsed[model] += metrics["elapsed"]
            sum_avg_llm[model] += metrics.get("llm", 0)

    for rev_data in best_rev_scores.values():
        for model, metrics in rev_data.items():
            sum_best_scores[model] += metrics["score"]
            sum_best_costs[model] += metrics["cost"]
            sum_best_elapsed[model] += metrics["elapsed"]
            sum_best_llm[model] += metrics.get("llm", 0)

    # Sort models for plotting. Groups are sorted by the best model in the
    # group. Within a group, models are sorted by their own score.
    model_groups: dict[str, list[str]] = {}
    for model_alias in models:
        raw_model_name = MODEL_MAPPING.get(model_alias, model_alias)
        model_groups.setdefault(raw_model_name, []).append(model_alias)

    group_max_scores = {
        raw_name: max(sum_avg_scores.get(alias, 0.0) for alias in aliases)
        for raw_name, aliases in model_groups.items()
    }

    sorted_group_names = sorted(
        model_groups.keys(), key=lambda g: group_max_scores[g], reverse=True
    )

    models = [
        alias
        for group_name in sorted_group_names
        for alias in sorted(
            model_groups[group_name],
            key=lambda m: sum_avg_scores.get(m, 0.0),
            reverse=True,
        )
    ]

    n_runs = len(complete_run_results)
    project_label = ", ".join(projects)
    base_dir_name = complete_run_results[0][1].parent.name

    if text_mode:
        # Flip axes: render horizontal ASCII bars instead of the vertical bar plot.
        avg_scores_percent_local = {m: (sum_avg_scores[m] / n_revisions) * 100.0 for m in models}
        best_scores_percent_local = {m: (sum_best_scores[m] / n_revisions) * 100.0 for m in models}
        _print_ascii_dual_series(
            title="Score",
            models=models,
            series1_name="Avg",
            series1_vals=avg_scores_percent_local,
            series2_name="Best",
            series2_vals=best_scores_percent_local,
            width=50,
            formatter=lambda v: f"{v:.2f}%",
            normalize_to_100=True,
        )
        
        # Compute pass rates per model
        pass_counts: dict[str, int] = {m: 0 for m in models}
        total_counts: dict[str, int] = {m: 0 for m in models}
        total_build_failures: dict[str, int] = {m: 0 for m in models}
        
        for comp_rev in all_revisions:
            for model in models:
                metrics_list = merged_rev_scores[comp_rev][model]
                for m in metrics_list:
                    total_counts[model] += 1
                    if m["score"] > 0:
                        pass_counts[model] += 1
                    total_build_failures[model] += m.get("build", 0)
        
        print("# Pass Count (build failures)")
        max_label = max((len(_pretty_name(m)) for m in models), default=0)
        bar_width = 50
        for m in models:
            display = _pretty_name(m)
            passes = int(pass_counts[m])
            total = int(total_counts[m])
            builds = int(total_build_failures[m])
            pass_rate = (passes / total * 100.0) if total > 0 else 0.0
            filled = int(round((pass_rate / 100.0) * bar_width))
            bar = "#" * max(0, min(filled, bar_width))
            print(f"{display:<{max_label}} | {bar:<{bar_width}} {passes:>2}/{total:<2} ({builds:>2}) {pass_rate:.1f}%")
        print("")
        # Per-revision plots are not rendered in text mode to keep output concise.
    else:
        if "mainscore" in charts_to_render:
            _plot(
                sum_avg_scores,
                sum_best_scores,
                sum_avg_costs,
                sum_best_costs,
                models,
                n_revisions,
                f"Projects: {project_label}",
                n_runs,
                base_dir_name,
                model_colors_global,
            )
        if "per_task_best" in charts_to_render:
            _plot_by_revision(
                best_rev_scores,
                models,
                title=f"Per-Task Best Scores (Projects: {project_label}, {n_runs} runs)",
                model_colors_global=model_colors_global,
            )
        if "per_task_avg" in charts_to_render:
            _plot_by_revision(
                avg_rev_scores,
                models,
                title=f"Per-Task Average Scores (Projects: {project_label}, {n_runs} runs)",
                model_colors_global=model_colors_global,
            )
        if "llm_runtime" in charts_to_render:
            _plot_llm_by_revision(
                merged_rev_scores,
                models,
                title="Per-Task LLM Runtime (Normalized)",
                model_colors_global=model_colors_global,
            )

    # Compute success-only spend and latency
    success_spend = _compute_success_only_spend(merged_rev_scores, models)
    success_latency = _compute_success_only_latency(merged_rev_scores, models)

    # Normalize scores to percentages for scatter plots
    avg_scores_percent = {
        m: (sum_avg_scores[m] / n_revisions) * 100 for m in models
    }
    best_scores_percent = {
        m: (sum_best_scores[m] / n_revisions) * 100 for m in models
    }

    if text_mode:
        # Represent scatter plots as bars of cost/time, with values at bar ends.
        _print_ascii_single_series(
            title=f"Cost (Successful Tasks, Projects: {project_label}, {n_runs} runs)",
            labels=models,
            values=success_spend,
            width=50,
            formatter=lambda c: f"${c:.2f}",
        )
        _print_ascii_single_series(
            title=f"Latency (Successful Tasks, Projects: {project_label}, {n_runs} runs)",
            labels=models,
            values=success_latency,
            width=50,
            formatter=_format_seconds,
        )
    else:
        if "score_v_spend" in charts_to_render:
            _plot_spend_scatter(
                success_spend=success_spend,
                y_data=avg_scores_percent,
                models=models,
                title="Average Score vs. Spend",
                model_colors_global=model_colors_global,
            )
        if "score_v_latency" in charts_to_render:
            _plot_latency_scatter(
                success_latency=success_latency,
                y_data=avg_scores_percent,
                models=models,
                title="Average Score vs. LLM Latency",
                model_colors_global=model_colors_global,
            )
        if "score_speed_price" in charts_to_render:
            # Score: use raw percentage as-is
            # Speed and Price: normalize as fraction of best (fastest/cheapest = 100%)
            norm_score = avg_scores_percent
            norm_speed = _normalize_as_fraction_of_best(success_latency)
            norm_price = _normalize_as_fraction_of_best(success_spend)
            _plot_radar(
                normalized_score=norm_score,
                normalized_speed=norm_speed,
                normalized_price=norm_price,
                models=models,
                title=f"Score vs. Speed vs. Price (Projects: {project_label}, {n_runs} runs)",
                model_colors_global=model_colors_global,
            )

    # ----------- Task-length bucketed plots -----------
    # Discover codetasks directories per project and load tokens for composite revisions.
    project_codetasks: dict[str, Path] = {}
    for project_name in projects:
        # Pick any run dir for this project to discover a nearby codetasks directory
        candidate_runs = [rd for (p, rd, _res) in complete_run_results if p == project_name]
        if not candidate_runs:
            continue
        base_dir = candidate_runs[0].parent
        candidates = [base_dir / "codetasks", base_dir.parent / "codetasks"]
        codetasks_dir = next((c for c in candidates if c.is_dir()), None)
        if codetasks_dir is not None:
            project_codetasks[project_name] = codetasks_dir

    tokens_by_rev: dict[str, int] = {}
    for comp_rev in all_revisions:
        if ":" in comp_rev:
            proj, rev = comp_rev.split(":", 1)
        else:
            # Fallback: assume first project if not prefixed (shouldn't happen)
            proj, rev = (projects[0] if projects else ""), comp_rev
        codetasks_dir = project_codetasks.get(proj)
        if not codetasks_dir:
            continue
        task_file = codetasks_dir / f"{rev}.json"
        if not task_file.is_file():
            continue
        try:
            data = json.loads(task_file.read_text())
            tokens_val = data.get("tokens")
            if isinstance(tokens_val, int) and tokens_val >= 0:
                tokens_by_rev[comp_rev] = tokens_val
        except Exception:
            continue

    if tokens_by_rev:
        avg_bucket_scores, avg_bucket_counts = _build_bucket_scores(avg_rev_scores, tokens_by_rev)
        best_bucket_scores, best_bucket_counts = _build_bucket_scores(best_rev_scores, tokens_by_rev)

        if text_mode:
            pass
        else:
            if "by_task_length_best" in charts_to_render:
                _plot_by_bucket(
                    best_bucket_scores,
                    best_bucket_counts,
                    models=models,
                    title=f"Best Scores by Task Length (Projects: {project_label}, {n_runs} runs)",
                    model_colors_global=model_colors_global,
                )
            if "by_task_length_avg" in charts_to_render:
                _plot_by_bucket(
                    avg_bucket_scores,
                    avg_bucket_counts,
                    models=models,
                    title=f"Average Scores by Task Length (Projects: {project_label}, {n_runs} runs)",
                    model_colors_global=model_colors_global,
                )
    else:
        print("Warning: Could not load any codetasks token data for the selected projects", file=sys.stderr)

    # ------------------- JSONL Exports -------------------
    avg_scores_percent = {
        m: (sum_avg_scores[m] / n_revisions) * 100 for m in models
    }
    sum_best_llm_in_seconds = {
        m: t / 1000.0 for m, t in sum_best_llm.items()
    }

    # 1) overall.jsonl — overall score aggregated by model
    # Compute total cost and latency (including all tasks, not just successful)
    total_spend = {m: 0.0 for m in models}
    total_latency = {m: 0.0 for m in models}
    for comp_rev in all_revisions:
        for model in models:
            metrics_list = merged_rev_scores[comp_rev][model]
            for metric in metrics_list:
                total_spend[model] += metric.get("cost", 0.0)
                llm_ms = metric.get("llm", 0.0)
                total_latency[model] += llm_ms / 1000.0
    
    overall_rows: list[dict] = []
    for m in models:
        overall_rows.append({
            "model": m,
            "n_tasks": n_revisions,
            "avg_score_percent": avg_scores_percent.get(m, 0.0),
            "best_score_percent": best_scores_percent.get(m, 0.0),
            "total_cost": float(total_spend.get(m, 0.0)),
            "success_cost": float(success_spend.get(m, 0.0)),
            "total_latency_seconds": float(total_latency.get(m, 0.0)),
            "success_latency_seconds": float(success_latency.get(m, 0.0)),
        })
    _write_jsonl(export_dir / "overall.jsonl", overall_rows)

    # 2) by_task.jsonl — aggregated by single task (per (task, model))
    by_task_rows: list[dict] = []
    for comp_rev in sorted(all_revisions):
        if ":" in comp_rev:
            proj, rev = comp_rev.split(":", 1)
        else:
            proj, rev = "", comp_rev
        tok = tokens_by_rev.get(comp_rev)
        avg_rev = avg_rev_scores.get(comp_rev, {})
        best_rev = best_rev_scores.get(comp_rev, {})
        for m in models:
            avg_metrics = avg_rev.get(m)
            best_metrics = best_rev.get(m)
            # Include entry only if any metrics exist
            if not avg_metrics and not best_metrics:
                continue

            row = {
                "project": proj,
                "revision": rev,
                "composite_revision": comp_rev,
                "model": m,
                "tokens": tok if tok is not None else None,
            }

            if avg_metrics:
                row.update({
                    "avg_score_percent": float(avg_metrics["score"] * 100.0),
                    "avg_cost": float(avg_metrics["cost"]),
                    "avg_elapsed_ms": float(avg_metrics["elapsed"]),
                    "avg_llm_ms": float(avg_metrics.get("llm", 0)),
                    "avg_retries": int(avg_metrics.get("retries", 0)),
                    "avg_build": int(avg_metrics.get("build", 0)),
                    "avg_parse": int(avg_metrics.get("parse", 0)),
                    "avg_apply": int(avg_metrics.get("apply", 0)),
                })
            else:
                row.update({
                    "avg_score_percent": None,
                    "avg_cost": None,
                    "avg_elapsed_ms": None,
                    "avg_llm_ms": None,
                    "avg_retries": None,
                    "avg_build": None,
                    "avg_parse": None,
                    "avg_apply": None,
                })

            if best_metrics:
                row.update({
                    "best_score_percent": float(best_metrics["score"] * 100.0),
                    "best_cost": float(best_metrics["cost"]),
                    "best_elapsed_ms": float(best_metrics["elapsed"]),
                    "best_llm_ms": float(best_metrics.get("llm", 0)),
                    "best_retries": int(best_metrics.get("retries", 0)),
                    "best_build": int(best_metrics.get("build", 0)),
                    "best_parse": int(best_metrics.get("parse", 0)),
                    "best_apply": int(best_metrics.get("apply", 0)),
                })
            else:
                row.update({
                    "best_score_percent": None,
                    "best_cost": None,
                    "best_elapsed_ms": None,
                    "best_llm_ms": None,
                    "best_retries": None,
                    "best_build": None,
                    "best_parse": None,
                    "best_apply": None,
                })

            by_task_rows.append(row)
    _write_jsonl(export_dir / "by_task.jsonl", by_task_rows)

    # 3) by_tokens.jsonl — aggregated by task tokens (length buckets)
    by_tokens_rows: list[dict] = []
    if tokens_by_rev:
        # Build a sorted list of bucket labels
        def _bucket_key(label: str) -> int:
            lp = label.split("-")[0]
            try:
                return int(lp)
            except ValueError:
                return 0

        for bucket in sorted(best_bucket_scores.keys(), key=_bucket_key):
            tasks_in_bucket = int(best_bucket_counts.get(bucket, 0))
            for m in models:
                avg_sum = float(avg_bucket_scores.get(bucket, {}).get(m, 0.0))
                best_sum = float(best_bucket_scores.get(bucket, {}).get(m, 0.0))
                avg_pct = (avg_sum / tasks_in_bucket) * 100.0 if tasks_in_bucket else 0.0
                best_pct = (best_sum / tasks_in_bucket) * 100.0 if tasks_in_bucket else 0.0
                by_tokens_rows.append({
                    "bucket": bucket,
                    "model": m,
                    "tasks_in_bucket": tasks_in_bucket,
                    "avg_score_percent": avg_pct,
                    "best_score_percent": best_pct,
                })
    _write_jsonl(export_dir / "by_tokens.jsonl", by_tokens_rows)

    # 4) score_vs_spend.jsonl — score vs spend
    score_vs_spend_rows: list[dict] = []
    for m in models:
        score_vs_spend_rows.append({
            "model": m,
            "success_spend": float(success_spend.get(m, 0.0)),
            "avg_score_percent": avg_scores_percent.get(m, 0.0),
            "best_score_percent": best_scores_percent.get(m, 0.0),
        })
    _write_jsonl(export_dir / "score_vs_spend.jsonl", score_vs_spend_rows)

    # 5) score_vs_speed.jsonl — score vs speed
    score_vs_speed_rows: list[dict] = []
    for m in models:
        score_vs_speed_rows.append({
            "model": m,
            "success_latency_seconds": float(success_latency.get(m, 0.0)),
            "avg_score_percent": avg_scores_percent.get(m, 0.0),
            "best_score_percent": best_scores_percent.get(m, 0.0),
        })
    _write_jsonl(export_dir / "score_vs_speed.jsonl", score_vs_speed_rows)

    if not text_mode:
        plt.show()


if __name__ == "__main__":
    main()
