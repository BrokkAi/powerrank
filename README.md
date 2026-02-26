# The Brokk Power Ranking
<img width="3213" height="1482" alt="image" src="https://github.com/user-attachments/assets/b14959dc-cdeb-4cf0-935b-6d6b657f4141" />

Full results at https://brokk.ai/power-rankings

In-depth analysis at https://blog.brokk.ai/introducing-the-brokk-power-ranking/

# Running the evaluation

## Prereqs

1. The test harness uses your local Brokk model definitions, so set up all the model aliases in the Favorite Models part of Settings. This is close to what I ended up with for the Open Round:
```
favoriteModelsJson=\[{"alias"\\:"o3","modelName"\\:"o3","reasoning"\\:"DEFAULT"},{"alias"\\:"gp2.5-default","modelName"\\:"gemini-2.5-pro","reasoning"\\:"DEFAULT"},{"alias"\\:"flash-2.5","modelName"\\:"gemini-2.5-flash","reasoning"\\:"DEFAULT"},{"alias"\\:"r1","modelName"\\:"deepseek-R1","reasoning"\\:"DEFAULT"},{"alias"\\:"gp 2.5 low","modelName"\\:"gemini-2.5-pro","reasoning"\\:"LOW"},{"alias"\\:"o3-high","modelName"\\:"o3","reasoning"\\:"HIGH"},{"alias"\\:"gp 2.5 high","modelName"\\:"gemini-2.5-pro","reasoning"\\:"HIGH"},{"alias"\\:"Flash 2.5 lite","modelName"\\:"gemini-2.5-flash-lite","reasoning"\\:"DEFAULT"},{"alias"\\:"o4-mini","modelName"\\:"o4-mini","reasoning"\\:"DEFAULT"},{"alias"\\:"Sonnet4","modelName"\\:"claude-4-sonnet","reasoning"\\:"MEDIUM"},{"alias"\\:"q3c","modelName"\\:"qwen3-coder","reasoning"\\:"DEFAULT"},{"alias"\\:"flash-2.5-nothink","modelName"\\:"gemini-2.5-flash","reasoning"\\:"DISABLE"},{"alias"\\:"sonnet3.7","modelName"\\:"claude-3.7-sonnet","reasoning"\\:"MEDIUM"},{"alias"\\:"gp2.5-high","modelName"\\:"gemini-2.5-pro","reasoning"\\:"HIGH"},{"alias"\\:"flash-2.0","modelName"\\:"gemini-2.0-flash","reasoning"\\:"DEFAULT"},{"alias"\\:"sonnet4-nothink","modelName"\\:"claude-4-sonnet","reasoning"\\:"DEFAULT"},{"alias"\\:"flash-2.5-high","modelName"\\:"gemini-2.5-flash","reasoning"\\:"HIGH"},{"alias"\\:"o4-mini-high","modelName"\\:"o4-mini","reasoning"\\:"HIGH"},{"alias"\\:"sonnet4-high","modelName"\\:"claude-4-sonnet","reasoning"\\:"HIGH"},{"alias"\\:"v3","modelName"\\:"deepseek-v3","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt-oss-120b","modelName"\\:"gpt-oss-120b","reasoning"\\:"DEFAULT"},{"alias"\\:"q3c-fp8","modelName"\\:"qwen3-coder","reasoning"\\:"DEFAULT"},{"alias"\\:"grok-3","modelName"\\:"grok-3-beta","reasoning"\\:"DEFAULT"},{"alias"\\:"k2","modelName"\\:"groq-kimi-k2","reasoning"\\:"DEFAULT"},{"alias"\\:"opus4.1","modelName"\\:"claude-4-1-opus","reasoning"\\:"MEDIUM"},{"alias"\\:"grok-3-mini","modelName"\\:"grok-3-mini-beta","reasoning"\\:"DEFAULT"},{"alias"\\:"grok-3-mini-high","modelName"\\:"grok-3-mini-beta","reasoning"\\:"HIGH"},{"alias"\\:"opus4.1-high","modelName"\\:"claude-4-1-opus","reasoning"\\:"HIGH"},{"alias"\\:"gpt-oss-120b-high","modelName"\\:"gpt-oss-120b","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt5","modelName"\\:"gpt-5","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt5-high","modelName"\\:"gpt-5","reasoning"\\:"HIGH"},{"alias"\\:"gpt5-mini","modelName"\\:"gpt-5-mini","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt5-mini-high","modelName"\\:"gpt-5-mini","reasoning"\\:"HIGH"},{"alias"\\:"gpt5-nano","modelName"\\:"gpt-5-nano","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt5-nano-high","modelName"\\:"gpt-5-nano","reasoning"\\:"HIGH"}\]
```
2. Clone and open each dataset project in Brokk first to populate Build configuration:
   1. `git@github.com:BrokkAi/brokk.git`
   2. `git@github.com:apache/kafka.git`
   3. `git@github.com:bitwarden/server.git`
   4. `git@github.com:astral-sh/ruff.git`
   5. `git@github.com:tailscale/tailscale.git`
3. Install support tooling for non-Java datasets:
   1. Rust toolchain  
   2. .NET toolchain (use docs for Debian/your OS)  
   3. `sccache`, `mold`, and `clang`
4. Configure `dataset_config.py` for per-project heap and command settings. Default behavior is centralized there; don’t hand-edit shell scripts for dataset-specific settings.
5. In `.env`, ensure proxy/block/model credentials are populated and match your Brokk proxy setup.

# Run the harness for each dataset.

These are the current default datasets in this workflow: `brokk`, `kafka`, `bitwarden`, `ruff`, and `tailscale`.

Start with a quicker sanity run before scaling out.
```
BROKK_PROXY=LOCALHOST ./bpr-all.sh --results-dir coderesults-0226 --runs 2 --model q3-35b --tui --threads 30
```

### Using a specific released CLI jar

If you downloaded a CLI jar from `https://github.com/BrokkAi/brokk-releases/releases` (for example `brokk-0.23.0.beta9.jar`), point the benchmark to it with `--cli-jar`.

```bash
./bpr-all.sh \
  --cli-dir . \
  --cli-jar "$(pwd)/brokk-0.23.0.beta9.jar" \
  --results-dir coderesults-0226 \
  --runs 2 \
  --model q3-35b \
  --tui \
  --threads 30
```

Equivalent direct `bpr.py` invocation:
CLI fallback behavior: if `--cli-jar` is omitted, `cli` uses the latest `brokk*.jar` from `./cli/app/build/libs`.

`bpr.py`/`bpr-all.sh` can now execute tasks from multiple projects concurrently, so separate per-project runs are no longer required. Use `--projects=...` with your full set when possible, and only split into smaller runs if you want manual resource isolation.

# How the sausage is made (explanation of bpr-all.sh)

* `BRK_TESTALL`: used by harness after the run to validate that everything passes. Only used for Brokk, everyone else has a test suite that’s too large to reasonably run for each task. (Also used by BrokkCli if no tests are present in the commit, but that should never be the case for our tasks.)  
* `BRK_TESTSOME`: used by cli CodeAgent to override project build config  
* `BRK_CPG_CACHE`: used by AnalyzerWrapper and Language to allow us to re-use the cpg across multiple runs (cache in $project/.brokk/cache is capped at 100\)  
* `--threads` controls concurrency both for your own machine (1GB heap per BrokkCli instance) and vendor quota  
  * Bitwarden and tailscale are heavier; ruff and Kafka can still be run with lower parallelism
* `BRK_NO_CONCURRENT_BUILDS`: enables a lock file so only one build runs per project at a time.

Notes on the datasets:

1. Brokk’s test suite is run in full after each edit.
2. Kafka is a multi-module Gradle project that needs test task filtering guardrails. The benchmark config injects `-I kafka-init.gradle` so `failOnNoMatchingTests=false` for modules with no matching test filter.
3. Ruff is a Rust monorepo; test selection uses a helper script that maps changed paths to crates for `cargo nextest`.
4. Tailscale uses its vendored Go toolchain in `./tool/go`; tests are wrapped to target only changed packages.
5. Bitwarden is maintained as a C# dataset and requires .NET tooling and typical C# dependency setup.

`run` will dump the results in the same directory as `generate`, `{mode}results/{project}{N}`.

# Checking for errors

## litellm
Litellm sometimes just gives up and says, “fuck you.” Need to clean these up and re-run. Inspect the results like this:  
`find coderesults-0226 -name "*.json" |xargs grep -l "check litellm"`  
Inspect the results, probably most will look like this:  
`{"elapsedMillis": 759249, "inputTokens": 2612321, "cachedInputTokens": 941696, "reasoningTokens": 83849, "outputTokens": 135477, "editBlocksTotal": 23, "editBlocksFailed": 9, "buildFailures": 2, "parseRetries": 1, "applyRetries": 8, "apiRetries": 0, "changedFiles": ["src/main/java/io/github/jbellis/brokk/gui/GitWorktreeTab.java", "src/main/java/io/github/jbellis/brokk/git/GitRepo.java"], "stopReason": "LLM_ERROR", "stopExplanation": "BadRequestError (no further information, the response was null; check litellm logs)", "worktree": "/Users/jonathan/brokkbench/brokk/flash-2.5-ddd83eaca-25-07-29-10-29"}`  
Remove them (`... |xargs rm`) and re-run the `run` script.

## Openrouter

1. Popular models may just be throttled to the point that they error out even after Brokk’s 8 exponential-backoff retries. Find these with  
   `find coderesults-0226 -name "*.json" |xargs grep -l "throttling_error"`  
2. Check for other random-ass errors by looking for `Openrouter` in the results, but filter out context-length errors which are expected when the model keeps trying right up until it can’t anymore:  
   `find coderesults-0226 -type f |xargs grep -l Openrouter |xargs grep -lv "context length"`

Again, remove the bad results and re-run them. (The harness will not run tasks for which results already exist.)

## In theory the `run` harness retries when it hits “check litellm” or “throttling\_error”

But you should doublecheck.

Also, the harness does NOT check for any other errors so you should definitely look.

## Other notes

* Some noisy test-log entries are usually harmless logging behavior for specific datasets; treat them as signals only when they correlate with failing tasks.

# Dealing with bugs in the `run` harness

1. Sometimes it gets stuck waiting on a subprocess. If this happens, interrupt with ctrl-c and restart.
2. Sometimes it fails to record metrics for a task, again I’m not sure why. The `results` script will warn you if you’re missing task entries and you can just re-run the `run` harness to fill in the blanks.

# Analysis and visualization

```
uv run results.py coderesults-0226 [--project brokk,kafka,bitwarden,ruff,tailscale] [--models o3,gp2.5-default] [--filter taskcommits-0126/openround.jsonl] [--no-estimate]
```

Lots of options in results.py, ask Brokk to explain them.
