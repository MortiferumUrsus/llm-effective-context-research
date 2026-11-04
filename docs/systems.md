# Systems and measurement scope

The study treats each procedure as a complete configured system. Prompts, tools, and call budgets differ. The results do not isolate the causal contribution of one memory component or rank entire method families.

## Full context

The baseline supplies the source and question to one model request. The historical runner uses an input-fit policy and approximate preflight token accounting. Recorded overflows are retained as unsuccessful attempts and excluded from conditional answer accuracy. The saved API token usage, rather than a universal words-to-tokens conversion, is used for workload figures.

## Hybrid retrieval

The historical procedure combines BM25 and BGE-small-en-v1.5 rankings by reciprocal rank fusion. It selects five 380-word chunks with 48-word overlap and presents them in retrieval rank order. A word-based chunk can exceed the encoder's 512-token window. A subsequent reconstruction matched all 900 saved BABILong selections to their previews. Complete reference-story fact-occurrence coverage averaged 16.42%; 126 selections contained no complete reference occurrence, and 888 differed from source order. These are not minimal supporting-fact recall measurements.

The inspection snapshot supports explicit source-order presentation and reports BM25-only operation if dense retrieval is unavailable. Those later options do not retroactively change the spring results.

## ReAct-style loop

The implementation uses document search, lookup, and a final-answer action, with up to six iterations and a forced-final fallback. Its tools operate over the fixed source rather than the original ReAct Wikipedia environment. Historical lookup repeatedly returned the first occurrence, making some temporal questions especially difficult. The available source includes a later cursor-based continuation fix. Scores in the report still come from the original saved responses.

## Memory-style loop

The implementation uses an archival BM25 search tool and a final-answer action. It retrieves chunks adaptively, avoids previously shown chunks, and adds task-specific instructions. It does not reproduce MemGPT's complete memory hierarchy or all of its tools. Its internal identifier `A_MemGPT` is preserved in data for traceability; prose uses **memory-style**.

The upstream document prompt is attributed to MemGPT tag `0.2.7`, commit `033d9d61f4e646193abe4f7699b1bf76c7f759f7`. That revision uses Apache-2.0. ReAct materials use the upstream MIT license. Notices are retained in the release.

## What the measurements mean

Source words describe the external material available to the system. Per-call model input includes instructions and conversation history. Input and output token totals for agent episodes sum their recorded model calls. These sums are not a measure of retrieval CPU time, embedding cost, or total monetary cost. Recorded durations are not consistently end-to-end timings and are not used to claim speedups.

The main empirical population is the selected spring API experiment. Local runs affected by serving-window limits are excluded from intrinsic-capacity interpretation. November checks are supplementary and separately dated. The main analysis never substitutes revised code for the code state that produced a saved answer.
