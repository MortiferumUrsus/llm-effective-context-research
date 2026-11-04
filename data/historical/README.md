# Historical observations

This directory contains a compact analysis export of the spring 2026 API experiment. The original raw logs are preserved separately. The file includes 9,800 observations selected by the final manifest; 9,506 are retained for conditional accuracy.

`observations.csv.gz` is a UTF-8 CSV compressed with gzip. It is deliberately an allowlisted table, not a dump of provider logs. `manifest.json` records its checksum, selection, and transformation.

| Fields | Meaning |
|---|---|
| `dataset_t`, `model`, `procedure_A` | Historical task, model alias, and implementation |
| `target_words`, `actual_words` | Nominal source-length tier and actual source word count |
| `seed`, `episode_idx`, `input_text_hash` | Sampling coordinates and the saved input fingerprint |
| `input_tokens`, `output_tokens` | Recorded usage; agent totals may span multiple model calls |
| `wall_clock_ms` | Historical recorded duration, not consistently end-to-end latency |
| `metric_value`, `metric_name` | Unchanged historical score and recorded metric label |
| `finish_reason`, `provider_fail` | Completion metadata used for availability and exclusions |
| `gold_answer`, `gold_provenance`, `output` | Reference answer, its provenance, and raw recorded answer text |
| `cluster_id`, `source_task_id`, `test_idx` | Recovered grouping identifiers for repeated evaluations |
| `timestamp`, `git_commit` | Historical recording metadata; commit alone does not reproduce dirty source state |

The historical `exact_match` label can describe substring scoring. The revised report explicitly distinguishes recorded scores from stricter answer-extraction sensitivity. Missing early NIAH gold answers are recovered only from hash-matched generator outputs, not from the model answers.

BABILong groups are derived from exact-prefix relations between reference stories; they are not verified original generator episode identifiers. Models are a fixed evaluated set. The call count must not be presented as the number of independently sampled problems.

The export excludes provider error bodies and account-management metadata. It does not include complete long source documents, every intermediate prompt, or a full record of historical provider routing. It reproduces the report's aggregate tables and figures, not every original model request.

`babilong_source_ids.json` maps saved input hashes to recovered source configurations, task IDs, and prefix-family IDs. `workload_evidence.csv` contains the per-call input/output counts for the 40 observations in the longest GPT-4o-mini NIAH worked example. The analysis verifies their sums against the main export.

The benchmarks retain their original attribution and terms. See [BABILong](https://arxiv.org/abs/2406.10149), the [Banking77 dataset card](https://huggingface.co/datasets/PolyAI/banking77), and the [third-party notices](../../THIRD_PARTY_NOTICES.md). No blanket license for third-party source corpora is asserted by this export.
