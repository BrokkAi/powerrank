# Running the evaluation

## Prereqs

1. The test harness uses your local Brokk model definitions, so set up all the model aliases in the Alternative Models part of Settings. This is close to what I ended up with for the Open Round:
```
favoriteModelsJson=\[{"alias"\\:"o3","modelName"\\:"o3","reasoning"\\:"DEFAULT"},{"alias"\\:"gp2.5-default","modelName"\\:"gemini-2.5-pro","reasoning"\\:"DEFAULT"},{"alias"\\:"flash-2.5","modelName"\\:"gemini-2.5-flash","reasoning"\\:"DEFAULT"},{"alias"\\:"r1","modelName"\\:"deepseek-R1","reasoning"\\:"DEFAULT"},{"alias"\\:"gp 2.5 low","modelName"\\:"gemini-2.5-pro","reasoning"\\:"LOW"},{"alias"\\:"o3-high","modelName"\\:"o3","reasoning"\\:"HIGH"},{"alias"\\:"gp 2.5 high","modelName"\\:"gemini-2.5-pro","reasoning"\\:"HIGH"},{"alias"\\:"Flash 2.5 lite","modelName"\\:"gemini-2.5-flash-lite","reasoning"\\:"DEFAULT"},{"alias"\\:"o4-mini","modelName"\\:"o4-mini","reasoning"\\:"DEFAULT"},{"alias"\\:"Sonnet4","modelName"\\:"claude-4-sonnet","reasoning"\\:"MEDIUM"},{"alias"\\:"q3c","modelName"\\:"qwen3-coder","reasoning"\\:"DEFAULT"},{"alias"\\:"flash-2.5-nothink","modelName"\\:"gemini-2.5-flash","reasoning"\\:"DISABLE"},{"alias"\\:"sonnet3.7","modelName"\\:"claude-3.7-sonnet","reasoning"\\:"MEDIUM"},{"alias"\\:"gp2.5-high","modelName"\\:"gemini-2.5-pro","reasoning"\\:"HIGH"},{"alias"\\:"flash-2.0","modelName"\\:"gemini-2.0-flash","reasoning"\\:"DEFAULT"},{"alias"\\:"sonnet4-nothink","modelName"\\:"claude-4-sonnet","reasoning"\\:"DEFAULT"},{"alias"\\:"flash-2.5-high","modelName"\\:"gemini-2.5-flash","reasoning"\\:"HIGH"},{"alias"\\:"o4-mini-high","modelName"\\:"o4-mini","reasoning"\\:"HIGH"},{"alias"\\:"sonnet4-high","modelName"\\:"claude-4-sonnet","reasoning"\\:"HIGH"},{"alias"\\:"v3","modelName"\\:"deepseek-v3","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt-oss-120b","modelName"\\:"gpt-oss-120b","reasoning"\\:"DEFAULT"},{"alias"\\:"q3c-fp8","modelName"\\:"qwen3-coder","reasoning"\\:"DEFAULT"},{"alias"\\:"grok-3","modelName"\\:"grok-3-beta","reasoning"\\:"DEFAULT"},{"alias"\\:"k2","modelName"\\:"groq-kimi-k2","reasoning"\\:"DEFAULT"},{"alias"\\:"opus4.1","modelName"\\:"claude-4-1-opus","reasoning"\\:"MEDIUM"},{"alias"\\:"grok-3-mini","modelName"\\:"grok-3-mini-beta","reasoning"\\:"DEFAULT"},{"alias"\\:"grok-3-mini-high","modelName"\\:"grok-3-mini-beta","reasoning"\\:"HIGH"},{"alias"\\:"opus4.1-high","modelName"\\:"claude-4-1-opus","reasoning"\\:"HIGH"},{"alias"\\:"gpt-oss-120b-high","modelName"\\:"gpt-oss-120b","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt5","modelName"\\:"gpt-5","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt5-high","modelName"\\:"gpt-5","reasoning"\\:"HIGH"},{"alias"\\:"gpt5-mini","modelName"\\:"gpt-5-mini","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt5-mini-high","modelName"\\:"gpt-5-mini","reasoning"\\:"HIGH"},{"alias"\\:"gpt5-nano","modelName"\\:"gpt-5-nano","reasoning"\\:"DEFAULT"},{"alias"\\:"gpt5-nano-high","modelName"\\:"gpt-5-nano","reasoning"\\:"HIGH"}\]
```
2. Clone and open each project in Brokk first to populate the Build configuration
   1. `git@github.com:apache/lucene.git`
   2. `git@github.com:BrokkAi/brokk.git`
   3. `git@github.com:eclipse-jgit/jgit.git`
   4. `git@github.com:datastax/cassandra.git`
      a. Apache's Cassandra repo uses git submodules now which the `run` script doesn't know how to deal with, so this targets DataStax's fork instead.
   5. `git@github.com:langchain4j/langchain4j.git`
3. JDKs (when in doubt, use sdkman)  
   1. Lucene requires exactly Java 24\.  
   2. Cassandra requires exactly Java 11\.  
   3. JGit requires \< 24, I used 21\.  
   4. LangChain4j and Brokk are less picky, I ran them with 24.

# Commands to run the harness for each dataset.

These include the full set of “open round” models for Brokk, and “finalist” models for the others. I recommend doing your first run with a couple faster/cheaper models (o4-mini and ds-v3 are my go-tos) as a sanity check before doing everything.

1. **Brokk**
```
   BRK\_TESTALL\_CMD='sbt test' BRK\_TESTSOME\_CMD='sbt "testOnly{{\#fqclasses}} {{value}}{{/fqclasses}}"' BRK\_CPG\_CACHE=true BB\_DEBUG=1 uv run run.py \--project ../brokk \--model o3,v3,gp2.5-default,gp2.5-high,o3-high,sonnet4,sonnet4-high,o4-mini,o4-mini-high,flash-2.0,flash-2.5,flash-2.5-nothink,flash-2.5-high,sonnet4-nothink,k2,q3c,r1,gpt-oss-120b,grok-3-mini,grok-3-mini-high,grok-3,opus4.1,q3c-fp8,opus4.1-high \--mode code \--runs 3 \--threads 50 \< results-6m/brokk.txt
```
2. **JGit**
```
   BRK\_SUPPRESS\_STDERR=true BRK\_TESTSOME\_CMD='JAVA\_HOME=/home/jonathan/.sdkman/candidates/java/21.0.8-amzn/ mvn \-DskipScriptExecution=true \--quiet test \-Dtest={{\#classes}}{{^first}},{{/first}}{{value}}{{/classes}} \-Dsurefire.failIfNoSpecifiedTests=false' BRK\_CPG\_CACHE=true BB\_DEBUG=1 uv run run.py \--project ../jgit \--model o3,gp2.5-default,sonnet4,o4-mini,flash-2.5,q3c,o4-mini-high,v3,opus4.1 \--mode code \--threads 50 \--runs 3 \< results-6m/jgit.txt
```
3. **LangChain4j**
```
   BRK\_TESTSOME\_CMD='JAVA\_HOME=/home/jonathan/.sdkman/candidates/java/21.0.8-amzn/ ./mvnw \--quiet test \-Dtest={{\#classes}}{{^first}},{{/first}}{{value}}{{/classes}} \-Dsurefire.failIfNoSpecifiedTests=false \-Dspotless.check.skip=true' BRK\_CPG\_CACHE=true BB\_DEBUG=1 uv run run.py \--project ../langchain4j \--model o3,flash-2.5,q3c,gp2.5-default,sonnet4,o4-mini,o4-mini-high,v3,opus4.1 \--mode code \--threads 50 \--runs 3 \< results-6m/langchain4j.txt
```
4. **Cassandra**
```
   BRK\_NO\_CONCURRENT\_BUILDS=true BRK\_TESTSOME\_CMD='JAVA\_HOME=/home/jonathan/.sdkman/candidates/java/11.0.28-amzn && {{\#classes}} ant test \-Dtest.name={{value}}{{^last}} && {{/last}} {{/classes}}' BRK\_CPG\_CACHE=true BB\_DEBUG=1 uv run run.py \-Xmx5G \--project ../cassandra-ds \--exclude CHANGES.txt \--model o3,flash-2.5,gp2.5-default,sonnet4,o4-mini,o4-mini-high,v3,q3c,opus4.1 \--mode code \--runs 3 \--threads 12 \< results-6m/cassandra.txt
```
5. **Lucene**
```
   BRK\_TESTSOME\_CMD='./gradlew \--quiet test{{\#classes}} \--tests {{value}}{{/classes}}' BRK\_CPG\_CACHE=true BB\_DEBUG=1 uv run run.py \-Xmx8G \--project ../lucene \--exclude lucene/CHANGES.txt \--model o3,flash-2.5,q3c,gp2.5-default,sonnet4,o4-mini,o4-mini-high,v3,opus4.1 \--mode code \--threads 8 \--runs 3 \< results-6m/lucene.txt
```

Explanation of these variables:

* `BRK_TESTALL`: used by harness after the run to validate that everything passes. Only used for Brokk, everyone else has a test suite that’s too large to reasonably run for each task. (Also used by BrokkCli if no tests are present in the commit, but that should never be the case for our tasks.)  
* `BRK_TESTSOME`: used by cli CodeAgent to override project build config  
* `BRK_CPG_CACHE`: used by AnalyzerWrapper and Language to allow us to re-use the cpg across multiple runs (cache in $project/.brokk/cache is capped at 100\)  
* `--threads` controls concurrency both for your own machine (1GB heap per BrokkCli instance) and vendor quota  
  * Cassandra and Lucene require larger heaps and we can thus run fewer concurrent threads  
* `BRK_NO_CONCURRENT_BUILDS`: enables a lock file to make sure only one build per project is happening at a time. This is **required** for Cassandra since some of its tests hardcode listening on a given local port and will error out if run concurrently. If you’re cranking up the concurrency you may want to add this flag to other datasets to reduce contention, but it shouldn’t be strictly necessary.

Notes on the datasets:

1. Brokk’s test suite is small enough to run the entire thing after each edit, so we do that.  
2. Langchain4j’s is small enough (under 2m on my MBA) to run once to make sure the agent didn’t break something outside its Workspace, but too slow to run for each edit.  
3. JGit’s test suite is full of broken tests for any given revision.  
4. Lucene (1h) and Cassandra’s (days) full test suites are impractical to run at all on a typical dev laptop

`run` will dump the results in the same directory as `generate`, `{mode}results/{project}{N}`.

## Checking for errors

### litellm
Litellm sometimes just gives up and says, “fuck you.” Need to clean these up and re-run. Inspect the results like this:  
`find coderesults -name "*.json" |xargs grep -l "check litellm"`  
Inspect the results, probably most will look like this:  
`{"elapsedMillis": 759249, "inputTokens": 2612321, "cachedInputTokens": 941696, "reasoningTokens": 83849, "outputTokens": 135477, "editBlocksTotal": 23, "editBlocksFailed": 9, "buildFailures": 2, "parseRetries": 1, "applyRetries": 8, "apiRetries": 0, "changedFiles": ["src/main/java/io/github/jbellis/brokk/gui/GitWorktreeTab.java", "src/main/java/io/github/jbellis/brokk/git/GitRepo.java"], "stopReason": "LLM_ERROR", "stopExplanation": "BadRequestError (no further information, the response was null; check litellm logs)", "worktree": "/Users/jonathan/brokkbench/brokk/flash-2.5-ddd83eaca-25-07-29-10-29"}`  
Remove them (`... |xargs rm`) and re-run the `run` script.

### Openrouter

1. Popular models may just be throttled to the point that they error out even after Brokk’s 8 exponential-backoff retries. Find these with  
   `find coderesults -name "*.json" |xargs grep -l "throttling_error"`  
2. Check for other random-ass errors by looking for `Openrouter` in the results, but filter out context-length errors which are expected when the model keeps trying right up until it can’t anymore:  
   `find coderesults -type f |xargs grep -l Openrouter |xargs grep -lv "context length"`

Again, remove the bad results and re-run them. (The harness will not run tasks for which results already exist.)

### In theory the `run` harness retries when it hits “check litellm” or “throttling\_error”

But you should doublecheck.

Also, the harness does NOT check for any other errors so you should definitely look.

### Other notes

* **T**he `[junit-timeout]` entries in Cassandra’s tests are NOT test timeouts, it’s some harmless logging thing.

## Dealing with bugs in the `run` harness

1. Sometimes it gets stuck waiting on a subprocess. I think this is specific to langchain4j’s test suite, and I also think (but I’m not completely sure) that I removed all the tasks that triggered it. Interrupt it with ctrl-c until it stops and restart it.  
2. Sometimes it fails to record metrics for a task, again I’m not sure why. The `results` script will warn you if you’re missing task entries and you can just re-run the `run` harness to fill in the blanks.

# Analysis and visualization

```
uv run results.py coderesults --project cassandra-ds,lucene,langchain4j,brokk,jgit --models o3,gp2.5-default
```