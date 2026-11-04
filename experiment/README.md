# Experiment implementation snapshot

This directory preserves the available core implementation: task loaders, the four processing procedures, scoring, API transport, and episode logging. It is provided for inspection alongside the saved-data analysis.

The source is a November 2026 snapshot. It includes later changes, notably explicit retrieval-order options and a ReAct lookup cursor. The exact spring dirty working tree was not fully captured. **Running this source would not constitute an exact replay of the reported spring experiment.** The supported reproduction entry point is `python -m analysis.final_report` from the repository root.

| Directory | Role |
|---|---|
| `bench/` | NIAH generation and BABILong/Banking77 task integration |
| `procedures/` | Full-context, RAG, ReAct-style, and memory-style functions |
| `llm/` | Model-client interface and historical OpenRouter adapter |
| `runner/` | Episode schema and append-only logging |
| `prompts/` | Attributed upstream reference prompt material |

`SNAPSHOT.json` records hashes before and after English documentation normalization. Changes in this copy are confined to comments and module documentation; executable statements and prompt strings are preserved. The workspace source itself was not modified by this export.

The four procedures are project adaptations with different tools, prompts, and budgets. The memory-style loop is not a complete MemGPT implementation. The historical ReAct result comes from the earlier lookup behavior, described in [Systems and measurement scope](../docs/systems.md). The snapshot also retains experimental choices, such as word-based chunking and approximate input preflight, which are not endorsed as current best practices.

Full service orchestration, automatic batch launchers, downloaded weights, credentials, and raw provider logs are not included. Source imports use the directory layout of the original harness. Inspecting or compiling these files is not a substitute for validating a new inference environment. This release makes no claim of having run the revised source against current model services.

The MemGPT document prompt is from tag `0.2.7` (Apache-2.0); ReAct reference examples retain their MIT notice. See [third-party attribution](../THIRD_PARTY_NOTICES.md).
