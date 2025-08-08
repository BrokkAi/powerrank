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


# Alias -> litellm model name
MODEL_MAPPING = {
    "o3": "o3",
    "o3-high": "o3",

    "gp2.5-default": "gemini/gemini-2.5-pro",
    "gp2.5-high": "gemini/gemini-2.5-pro",

    "v3": "deepseek/deepseek-v3",

    "o4-mini": "o4-mini",
    "o4-mini-high": "o4-mini",

    "sonnet4-nothink": "claude-4-sonnet-20250514",
    "sonnet4": "claude-4-sonnet-20250514",
    "sonnet4-high": "claude-4-sonnet-20250514",

    "flash-2.0": "gemini/gemini-2.0-flash",

    "flash-2.5": "gemini/gemini-2.5-flash",
    "flash-2.5-high": "gemini/gemini-2.5-flash",
    "flash-2.5-nothink": "gemini/gemini-2.5-flash",
}

# Custom per-token pricing for models not recognised by LiteLLM
CUSTOM_MODEL_PRICING: dict[str, CostPerToken] = {
    "k2":           {"input_cost_per_token": 1e-06,  "output_cost_per_token": 3e-06},
    "gpt-oss-120b": {"input_cost_per_token": 1.5e-07,"output_cost_per_token": 6e-07},
    "grok-3":       {"input_cost_per_token": 3e-06,  "output_cost_per_token": 1.5e-6},
    "grok-3-mini":  {"input_cost_per_token": 3e-07,  "output_cost_per_token": 5e-7},
    "grok-3-mini-high":  {"input_cost_per_token": 3e-07,  "output_cost_per_token": 5e-7},
    "opus4.1":      {"input_cost_per_token": 1.5e-05,"output_cost_per_token": 7.5e-5},
    "opus4.1-high": {"input_cost_per_token": 1.5e-05,"output_cost_per_token": 7.5e-5},
    "q3c":          {"input_cost_per_token": 1e-06,  "output_cost_per_token": 5e-06},
    "q3c-fp8":      {"input_cost_per_token": 4e-07,  "output_cost_per_token": 1.6e-06},
    "r1":           {"input_cost_per_token": 5.5e-07,  "output_cost_per_token": 2.19e-06},
    "gpt5":         {"input_cost_per_token": 1.25e-06,"output_cost_per_token": 1e-05},
    "gpt5-mini":    {"input_cost_per_token": 2.5e-07,"output_cost_per_token": 2e-06},
    "gpt5-nano":    {"input_cost_per_token": 5e-08,"output_cost_per_token": 4e-07},
}


# Color-blind-friendly palette (Okabe–Ito)
_COLORBLIND_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#009E73",  # bluish-green
    "#CC79A7",  # reddish-purple
    "#56B4E9",  # sky-blue
    "#E69F00",  # orange
    "#000000",  # black
]


def _get_distinct_colors(n: int) -> list[str]:
    """
    Return `n` visually distinct, colour-blind-friendly colours.
    Cycles the 8-colour Okabe–Ito palette as required.
    """
    repeats = (n + len(_COLORBLIND_PALETTE) - 1) // len(_COLORBLIND_PALETTE)
    return (_COLORBLIND_PALETTE * repeats)[:n]


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


def _parse_args() -> tuple[list[tuple[str, Path]], set[str] | None, set[str] | None, list[str]]:
    """
    Parse command-line arguments.

    Returns
    -------
    tuple
        (run_info, models, exclude_models, projects) where:
          - run_info is a list of (project, Path) tuples, one per run directory
          - models is a set of model names supplied via ``--models`` or ``None``
          - exclude_models is a set of model names supplied via ``--exclude`` or ``None``
          - projects is the list of project name prefixes
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
    args = parser.parse_args()

    if not args.base_directory.is_dir():
        parser.error(f"{args.base_directory} is not a directory")

    projects = [p.strip() for p in str(args.project).split(",") if p.strip()]
    if not projects:
        parser.error("--project must be a comma-separated list of one or more project names.")

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

    models: set[str] | None = None
    if args.models:
        models = {m.strip() for m in args.models.split(",") if m.strip()}
        if not models:
            parser.error("--models provided but no valid model names parsed")

    exclude_models: set[str] | None = None
    if args.exclude:
        exclude_models = {m.strip() for m in args.exclude.split(",") if m.strip()}
        if not exclude_models:
            parser.error("--exclude provided but no valid model names parsed")

    return run_info, models, exclude_models, projects


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
):
    """
    Read all JSON files inside `directory` and aggregate information per model.

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
    set of revision files as every other model in the directory.
    """
    scores: dict[str, float] = {}
    elapsed: dict[str, int] = {}
    costs: dict[str, float] = {}
    revisions_by_model: dict[str, set[str]] = {}
    rev_scores: dict[str, dict[str, dict]] = {}

    found_models: set[str] = set()  # Models that passed all filters and were processed

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

        # Track revisions encountered for each model
        revisions_by_model.setdefault(model, set()).add(revision)

        try:
            data = json.loads(json_file.read_text())

            # Compute cost for this revision irrespective of success/failure
            cost = 0.0
            response_for_cost = {
                "usage": {
                    "prompt_tokens": data.get("inputTokens", 0),
                    "completion_tokens": data.get("outputTokens", 0),
                    "cache_read_input_tokens": data.get("cachedInputTokens", 0),
                }
            }

            # Prefer custom pricing if defined for this model
            cost = 0
            if model in CUSTOM_MODEL_PRICING:
                custom_prices = CUSTOM_MODEL_PRICING[model]
                cost = completion_cost(
                    completion_response=response_for_cost,
                    model=model,  # model may be unknown to LiteLLM
                    custom_cost_per_token=custom_prices,
                    custom_pricing=True,
                )
            else:
                litellm_model = MODEL_MAPPING.get(model)
                if litellm_model:
                    cost = completion_cost(
                        completion_response=response_for_cost,
                        model=litellm_model,
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

    # If an 'include' filter was provided, ensure all requested models (that weren't excluded) were found
    if include_models is not None:
        # Effective included models are those explicitly included, minus any explicitly excluded
        effective_included_models = include_models - (exclude_models or set())
        # Check if any effectively included models were not found
        missing = effective_included_models - found_models
        if missing:
            raise ValueError(
                f"Requested models (after exclusions) not found: {', '.join(sorted(missing))}"
            )

    # Ensure every model has the exact same set of revisions
    all_revisions = set.union(*revisions_by_model.values())
    for model, revs in revisions_by_model.items():
        if revs != all_revisions:
            missing = all_revisions - revs
            raise ValueError(
                f"Inconsistent revisions detected: model '{model}' is missing "
                f"revisions {sorted(missing)}"
            )

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


def _plot(
    sum_avg_scores: dict[str, float],
    sum_best_scores: dict[str, float],
    sum_avg_costs: dict[str, float],
    sum_best_costs: dict[str, float],
    models: list[str],
    n_revisions: int,
    project_name: str,
    n_runs: int,
) -> None:
    if not models:
        print("No successful benchmark results found.", file=sys.stderr)
        sys.exit(1)

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 1 + 1.2 * len(models)), 7))
    fig.canvas.mpl_connect('key_press_event', _on_key_press)

    # Normalize scores to percentages
    avg_scores_vals = [
        (sum_avg_scores[m] / n_revisions) * 100 for m in models
    ]
    best_scores_vals = [
        (sum_best_scores[m] / n_revisions) * 100 for m in models
    ]

    colors = _get_distinct_colors(2)
    rects1 = ax.bar(
        x - width / 2, avg_scores_vals, width, label="Average Score", color=colors[0]
    )
    rects2 = ax.bar(
        x + width / 2, best_scores_vals, width, label="Best Score", color=colors[1]
    )

    ax.set_ylabel("Average Score (%)")
    ax.set_title(f"Average Scores for {project_name} ({n_runs} runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 100) # Scores are now percentages
    ax.set_yticks(np.arange(0, 101, 10)) # Set y-ticks from 0 to 100 in steps of 10
    # No horizontal line needed at n_revisions anymore as scale is 0-100%

    # Annotate bars
    for i, model in enumerate(models):
        # Avg bar
        h_avg = rects1[i].get_height()
        ax.text(
            rects1[i].get_x() + rects1[i].get_width() / 2,
            h_avg,
            f"{h_avg:.2f}%", # Display as percentage
            ha="center",
            va="bottom",
            fontsize=9,
        )

        # Best bar
        h_best = rects2[i].get_height()
        ax.text(
            rects2[i].get_x() + rects2[i].get_width() / 2,
            h_best,
            f"{h_best:.2f}%", # Display as percentage
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()


def _plot_by_revision(
    rev_scores: dict[str, dict[str, dict]], models: list[str], title: str
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
    revision_labels = [rev[:7] for rev in revisions]

    n_revisions = len(revisions)
    n_models = len(models)

    group_height = 0.8              # vertical fraction used by all bars in a group
    bar_height = group_height / n_models

    fig_width = 12
    # 50 % taller for better readability
    fig_height = max(6, 0.675 * n_revisions)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.canvas.mpl_connect('key_press_event', _on_key_press)

    color_list = _get_distinct_colors(n_models)
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
        plt.matplotlib.patches.Patch(color=cmap(i), label=model)
        for i, model in enumerate(models)
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
) -> None:
    """
    Draw a scatter plot with labels for each point.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.mpl_connect("key_press_event", _on_key_press)

    n_models = len(models)
    color_list = _get_distinct_colors(n_models)
    cmap = mcolors.ListedColormap(color_list)

    for i, model in enumerate(models):
        x_val = x_data.get(model, 0.0)
        y_val = y_data.get(model, 0.0)

        ax.scatter(x_val, y_val, color=cmap(i))
        ax.text(x_val, y_val, f"  {model}", fontsize=9, va="center")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_ylim(bottom=0)
    if x_log_base:
        ax.set_xscale("log", base=x_log_base)
    else:
        ax.set_xlim(left=0)

    legend_handles = [
        plt.matplotlib.patches.Patch(color=cmap(i), label=model)
        for i, model in enumerate(models)
    ]
    ax.legend(handles=legend_handles, title="Models", loc="best", fontsize=8)

    plt.tight_layout()


def _format_seconds(seconds: float) -> str:
    """Convert seconds to a concise "XmYs" representation."""
    if seconds >= 60:
        minutes = int(seconds) // 60
        secs = int(round(seconds - minutes * 60))
        return f"{minutes}m{secs}s"
    return f"{seconds:.1f}s" if seconds < 10 else f"{int(round(seconds))}s"


def _plot_dual_metric_scatter(
    total_x_data: dict[str, float],
    success_x_data: dict[str, float],
    y_data: dict[str, float],
    models: list[str],
    title: str,
    x_label: str,
    x_formatter: Callable,
    x_axis_formatter: Callable | None = None,
) -> None:
    """
    Generic scatter plot for dual metrics (e.g., total vs success-only).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.mpl_connect("key_press_event", _on_key_press)

    n_models = len(models)
    cmap = mcolors.ListedColormap(_get_distinct_colors(n_models))

    for i, model in enumerate(models):
        total_x_val = total_x_data.get(model, 0.0)
        success_x_val = success_x_data.get(model, 0.0)
        y_val = y_data.get(model, 0.0)

        # Filled marker for success-only
        ax.scatter(success_x_val, y_val, color=cmap(i), s=40, zorder=3)
        # Hollow, larger marker for total
        ax.scatter(
            total_x_val,
            y_val,
            facecolors="none",
            edgecolors=cmap(i),
            linewidths=1.5,
            s=100,
            zorder=2,
        )

        # Annotations
        ax.text(
            success_x_val,
            y_val,
            f"  {model}\n  {x_formatter(success_x_val)}",
            fontsize=8,
            va="center",
        )
        ax.text(
            total_x_val,
            y_val,
            f"  {model}\n  {x_formatter(total_x_val)}",
            fontsize=8,
            va="center",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    if x_axis_formatter:
        ax.xaxis.set_major_formatter(x_axis_formatter)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)

    # Legend explaining marker styles
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="black", markerfacecolor="none",
            markeredgewidth=1.5, markersize=10, linestyle="", label="All tasks"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="black", markerfacecolor="black",
            markersize=6, linestyle="", label="Successful tasks"
        ),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8)

    plt.tight_layout()


def _plot_latency_scatter(
    avg_latency: dict[str, float],
    success_latency: dict[str, float],
    y_data: dict[str, float],
    models: list[str],
    title: str,
) -> None:
    """
    Scatter plot displaying two latency metrics per model:
    • Large hollow circle – total latency (average across runs, all tasks)
    • Filled circle      – latency for successful tasks only
    """
    _plot_dual_metric_scatter(
        total_x_data=avg_latency,
        success_x_data=success_latency,
        y_data=y_data,
        models=models,
        title=title,
        x_label="Total LLM Latency (s)",
        x_formatter=_format_seconds,
        x_axis_formatter=mticker.FuncFormatter(lambda x, _p: f"{x:,.0f}"),
    )


def _plot_spend_scatter(
    total_spend: dict[str, float],
    success_spend: dict[str, float],
    y_data: dict[str, float],
    models: list[str],
    title: str,
) -> None:
    """
    Scatter plot displaying two spend metrics per model:
    • Large hollow circle – total spend (average across runs, all tasks)
    • Filled circle      – spend for successful tasks only
    """
    _plot_dual_metric_scatter(
        total_x_data=total_spend,
        success_x_data=success_spend,
        y_data=y_data,
        models=models,
        title=title,
        x_label="Total Spend ($)",
        x_formatter=lambda cost: f"${cost:.2f}",
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


def _plot_by_bucket(
    bucket_scores: dict[str, dict[str, float]],
    bucket_task_counts: dict[str, int],
    models: list[str],
    title: str,
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

    color_list = _get_distinct_colors(n_models)
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


def main() -> None:
    run_info, models_filter, exclude_models_filter, projects = _parse_args()

    run_results: list[tuple[str, tuple]] = []
    for project_name, run_dir in run_info:
        try:
            run_results.append((
                project_name,
                _aggregate(run_dir, models_filter, exclude_models_filter)
            ))
        except ValueError as e:
            print(
                f"Skipping run {run_dir.name} due to error: {e}", file=sys.stderr
            )
            continue

    if not run_results:
        print("No valid run data found.", file=sys.stderr)
        sys.exit(1)

    # Determine export directory under the base directory of runs
    export_dir = run_info[0][1].parent / "jsonexport"

    # Collect per-run rev_scores along with their project
    runs_rev_scores: list[tuple[str, dict[str, dict[str, dict]]]] = [
        (project, res[4]) for (project, res) in run_results
    ]

    all_models: set[str] = set()
    all_revisions: set[str] = set()  # composite "<project>:<revision>"
    for project_name, rev_scores in runs_rev_scores:
        for rev, model_data in rev_scores.items():
            all_revisions.add(f"{project_name}:{rev}")
            all_models.update(model_data.keys())

    models = sorted(list(all_models))
    n_revisions = len(all_revisions)

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

    n_runs = len(run_info)
    project_label = ", ".join(projects)

    _plot(
        sum_avg_scores,
        sum_best_scores,
        sum_avg_costs,
        sum_best_costs,
        models,
        n_revisions,
        f"Projects: {project_label}",
        n_runs,
    )
    _plot_by_revision(
        best_rev_scores,
        models,
        title=f"Per-Task Best Scores (Projects: {project_label}, {n_runs} runs)",
    )
    _plot_by_revision(
        avg_rev_scores,
        models,
        title=f"Per-Task Average Scores (Projects: {project_label}, {n_runs} runs)",
    )

    sum_avg_llm_in_seconds = {
        m: t / 1000.0 for m, t in sum_avg_llm.items()
    }

    # Sum of averaged-per-task latencies ONLY for successful runs
    sum_success_llm_in_seconds = {m: 0.0 for m in models}
    for rev in all_revisions:
        for model in models:
            metrics_list = merged_rev_scores[rev][model]
            success_llms = [m["llm"] for m in metrics_list if m["score"] > 0]
            if success_llms:
                avg_success_llm = sum(success_llms) / len(success_llms)
                sum_success_llm_in_seconds[model] += avg_success_llm / 1000.0

    # Sum of averaged-per-task costs ONLY for successful runs
    sum_success_costs = {m: 0.0 for m in models}
    for rev in all_revisions:
        for model in models:
            metrics_list = merged_rev_scores[rev][model]
            success_costs = [m["cost"] for m in metrics_list if m["score"] > 0]
            if success_costs:
                avg_success_cost = sum(success_costs) / len(success_costs)
                sum_success_costs[model] += avg_success_cost

    # Normalize best scores to percentage for scatter plots
    best_scores_percent = {
        m: (sum_best_scores[m] / n_revisions) * 100 for m in models
    }

    _plot_spend_scatter(
        total_spend=sum_avg_costs,
        success_spend=sum_success_costs,
        y_data=best_scores_percent,
        models=models,
        title=f"Best Score vs. Spend (Projects: {project_label}, {n_runs} runs)",
    )
    _plot_latency_scatter(
        avg_latency=sum_avg_llm_in_seconds,
        success_latency=sum_success_llm_in_seconds,
        y_data=best_scores_percent,
        models=models,
        title=f"Best Score vs. LLM Latency (Projects: {project_label}, {n_runs} runs)",
    )

    # ----------- Task-length bucketed plots -----------
    # Discover codetasks directories per project and load tokens for composite revisions.
    project_codetasks: dict[str, Path] = {}
    for project_name in projects:
        # Pick any run dir for this project to discover a nearby codetasks directory
        candidate_runs = [rd for (p, rd) in run_info if p == project_name]
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

        _plot_by_bucket(
            best_bucket_scores,
            best_bucket_counts,
            models=models,
            title=f"Best Scores by Task Length (Projects: {project_label}, {n_runs} runs)",
        )
        _plot_by_bucket(
            avg_bucket_scores,
            avg_bucket_counts,
            models=models,
            title=f"Average Scores by Task Length (Projects: {project_label}, {n_runs} runs)",
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
    overall_rows: list[dict] = []
    for m in models:
        overall_rows.append({
            "model": m,
            "n_tasks": n_revisions,
            "avg_score_percent": avg_scores_percent.get(m, 0.0),
            "best_score_percent": best_scores_percent.get(m, 0.0),
            "sum_avg_cost": float(sum_avg_costs.get(m, 0.0)),
            "sum_best_cost": float(sum_best_costs.get(m, 0.0)),
            "sum_avg_latency_seconds": float(sum_avg_llm_in_seconds.get(m, 0.0)),
            "sum_best_latency_seconds": float(sum_best_llm_in_seconds.get(m, 0.0)),
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
            "total_spend": float(sum_avg_costs.get(m, 0.0)),
            "success_spend": float(sum_success_costs.get(m, 0.0)),
            "avg_score_percent": avg_scores_percent.get(m, 0.0),
            "best_score_percent": best_scores_percent.get(m, 0.0),
        })
    _write_jsonl(export_dir / "score_vs_spend.jsonl", score_vs_spend_rows)

    # 5) score_vs_speed.jsonl — score vs speed
    score_vs_speed_rows: list[dict] = []
    for m in models:
        score_vs_speed_rows.append({
            "model": m,
            "total_latency_seconds": float(sum_avg_llm_in_seconds.get(m, 0.0)),
            "success_latency_seconds": float(sum_success_llm_in_seconds.get(m, 0.0)),
            "avg_score_percent": avg_scores_percent.get(m, 0.0),
            "best_score_percent": best_scores_percent.get(m, 0.0),
        })
    _write_jsonl(export_dir / "score_vs_speed.jsonl", score_vs_speed_rows)

    plt.show()


if __name__ == "__main__":
    main()
