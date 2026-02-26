#!/usr/bin/env bash
# bpr-all.sh: Multi-project benchmark orchestrator
#
# Run benchmarks with --model across all projects
#
# Usage:
#   BROKK_PROXY=LOCALHOST ./bpr-all.sh --results-dir coderesults-0226 --runs 2 \
#                --model q3-35b --tui --threads 30
#   ./bpr-all.sh --projects brokk,langchain4j --model gem3flash --runs 3 --results-dir coderesults-0226
#
# API keys are read from .env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
MODEL=""
PLAN_MODEL=""
PROJECTS=""
RUNS=3
THREADS=""
MAXHEAP=""
MODE="code"
RESULTS_DIR=""
COMMITS_DIR="${BRK_COMMITS_DIR:-taskcommits-0126}"
DRY_RUN=""
USE_TUI=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --projects=*) PROJECTS="${1#*=}"; shift ;;
        --projects) PROJECTS="$2"; shift 2 ;;
        --tui) USE_TUI="1"; shift ;;
        --runs=*) RUNS="${1#*=}"; shift ;;
        --runs) RUNS="$2"; shift 2 ;;
        --threads=*) THREADS="${1#*=}"; shift ;;
        --threads) THREADS="$2"; shift 2 ;;
        --maxheap=*) MAXHEAP="${1#*=}"; shift ;;
        --maxheap) MAXHEAP="$2"; shift 2 ;;
        --infer-context) MODE="infer-context"; shift ;;
        --mode=*) MODE="${1#*=}"; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --results-dir=*) RESULTS_DIR="${1#*=}"; shift ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --commits-dir=*) COMMITS_DIR="${1#*=}"; shift ;;
        --commits-dir) COMMITS_DIR="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --planmodel=*) PLAN_MODEL="${1#*=}"; shift ;;
        --planmodel) PLAN_MODEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model required" >&2
    exit 1
fi

# Source .env and save API keys
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

_SAVED_OPENAI_KEY="${OPENAI_API_KEY:-}"
_SAVED_GEMINI_KEY="${GEMINI_API_KEY:-}"
_SAVED_DEEPSEEK_KEY="${DEEPSEEK_API_KEY:-}"

# Restore all API keys (for bpr.py which doesn't use client.py)
restore_all_keys() {
    [[ -n "$_SAVED_OPENAI_KEY" ]] && export OPENAI_API_KEY="$_SAVED_OPENAI_KEY" || true
    [[ -n "$_SAVED_GEMINI_KEY" ]] && export GEMINI_API_KEY="$_SAVED_GEMINI_KEY" || true
    [[ -n "$_SAVED_DEEPSEEK_KEY" ]] && export DEEPSEEK_API_KEY="$_SAVED_DEEPSEEK_KEY" || true
}

# Build common bpr.py args
BPR_ARGS=(--mode="$MODE" --runs="$RUNS" --commits-dir="$COMMITS_DIR")
[[ -n "$PLAN_MODEL" ]] && BPR_ARGS+=(--planmodel="$PLAN_MODEL") || true
[[ -n "$PROJECTS" ]] && BPR_ARGS+=(--projects="$PROJECTS") || true
[[ -n "$THREADS" ]] && BPR_ARGS+=(--threads="$THREADS") || true
[[ -n "$MAXHEAP" ]] && BPR_ARGS+=(--maxheap="$MAXHEAP") || true
[[ -n "$RESULTS_DIR" ]] && BPR_ARGS+=(--results-dir="$RESULTS_DIR") || true
[[ -n "$DRY_RUN" ]] && BPR_ARGS+=("$DRY_RUN") || true
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && BPR_ARGS+=("${EXTRA_ARGS[@]}") || true

# Determine results root (mirrors bpr.py logic)
if [[ -n "$RESULTS_DIR" ]]; then
    RESULTS_ROOT="$RESULTS_DIR"
else
    RESULTS_ROOT="${MODE}results"
fi

# Split projects into array (if provided, otherwise inferred for Phase 5)
if [[ -n "$PROJECTS" ]]; then
    IFS=',' read -ra PROJECT_LIST <<< "$PROJECTS"
else
    # Inferred from directories in results root if PROJECTS is empty
    PROJECT_LIST=()
fi

echo "========================================" >&2
echo "Benchmark Workflow" >&2
echo "  Model: ${MODEL}" >&2
[[ -n "$PLAN_MODEL" ]] && echo "  Plan Model: ${PLAN_MODEL}" >&2 || true
echo "  Projects: ${PROJECTS:-all}" >&2
echo "  Runs: ${RUNS}" >&2
[[ -n "$THREADS" ]] && echo "  Threads: ${THREADS}" >&2 || true
[[ -n "$MAXHEAP" ]] && echo "  Max heap: ${MAXHEAP}MB" >&2 || true
echo "  Results: ${RESULTS_ROOT}" >&2
echo "========================================" >&2

# Benchmark mode: Single model run
echo "" >&2
echo "=== Benchmark Mode: ${MODEL} ===" >&2
restore_all_keys
TARGET_SCRIPT="bpr.py"
if [[ -n "$USE_TUI" ]]; then
    TARGET_SCRIPT="bpr_tui.py"
fi

uv run "$TARGET_SCRIPT" --model="$MODEL" "${BPR_ARGS[@]}" || true
echo "=== Benchmark complete ===" >&2

echo "" >&2
echo "========================================" >&2
echo "Benchmark complete for: ${MODEL}" >&2
echo "========================================" >&2
