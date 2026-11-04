# Reproducing the report

The supported workflow analyzes the included observations. It makes no model calls, downloads no weights, and requires no credentials. The main observations date from spring 2026.

## Environment

Use Python 3.11 and an isolated environment:

```text
python -m venv .venv-analysis
```

Activate it with `.venv-analysis\Scripts\Activate.ps1` in PowerShell or `source .venv-analysis/bin/activate` on macOS/Linux. Then:

```text
python -m pip install -r requirements-analysis-lock.txt
python -m analysis.final_report --out results/reproduced
python -m unittest discover -s tests -p test_final_report.py -v
```

The lock file records the tested package versions. The input checksum is checked against `data/historical/manifest.json`. The public package also includes the recovered BABILong source mapping and 40 records of per-call token accounting for the worked example.

## Outputs and interpretation

| File in the output directory | Contents |
|---|---|
| `context_profiles.csv` | Accuracy, nominal Wilson intervals, actual source length, input and output tokens per condition |
| `availability_and_accuracy.csv` | Attempts, retained observations, conditional accuracy, and success per attempt |
| `paired_cluster_differences.csv` | Same-input method differences with task-cluster sensitivity intervals |
| `matched_length_differences.csv` | Task-matched BABILong source-configuration differences |
| `matched_length_pairs.csv` | Individual units behind those differences |
| `babilong_collapsed_conditions.csv` | Source conditions after averaging repeated identical inputs |
| `common_task_panel.csv` | Eight source tasks present in all three BABILong configurations |
| `observed_threshold_patterns.csv` | Descriptive 50% pattern labels, explicitly not capacity estimates |
| `model_task_accuracy.csv` | Exact values behind the complete model-task matrix |
| `workload_example.csv` | Verified per-episode workload for the 40-record worked example |
| `answer_sensitivity.csv` | Historical scoring versus a fixed conservative answer rule |
| `analysis_manifest.json` | Scope, checksums, dates, seeds, and interval interpretation |
| `figures/` | Four figures in PNG and SVG formats |

The frozen earlier analysis also produces `observed_cells.csv`, whose subset of columns is retained for compatibility. No fitted or extrapolated capacity values are generated.

## Statistical specification

Historical answer scores remain primary. Provider failures and context overflows leave the conditional-accuracy population but remain unsuccessful attempts for operational success. Agent answer-format failures stay in the population. Method pairs must agree on input hash, gold answer, and task group.

Wilson intervals are nominal single-cell intervals. Aggregate method comparisons resample whole task groups 20,000 times using seed 20261102; known repeated conditions stay together and models are not resampled. Prefix-related BABILong stories form derived sensitivity groups, not proven independent episodes. Degenerate NIAH bootstrap intervals must not be interpreted as certainty.

For length comparisons, `source_config` comes from the recovered input-hash mapping. The unit is model, source task, and source configuration. Repeated answers are averaged only after verifying identical input hashes, gold answers, and actual lengths. Conditions are paired by model, source task, and prefix family. Their difference is longer-source accuracy minus shorter-source accuracy. The bootstrap uses lexicographically ordered prefix-family IDs, 20,000 draws, and a fresh generator with seed 20261102 for each comparison. Each draw divides the sum of differences by the number of task-model units sampled. The fixed model set is retained. Averaged responses can be fractional, so a binary McNemar test is not applied to these contrasts.

The three pairwise intersections contain different tasks. Background text also changes between source configurations. These are exploratory matched-task comparisons, not isolated causal effects of length.

## Rebuilding the PDF

The PDF is generated from `report.md` and the saved PNG figures:

```text
python -m pip install -r requirements-pdf.txt
python scripts/build_report_pdf.py
```

The output is `output/pdf/effective_context_report.pdf`. A PDF viewer is sufficient to read it; rebuilding the analysis does not require PDF dependencies. Page images were inspected during release preparation. The renderer embeds the Bitstream Vera fonts bundled with ReportLab, preserving text spacing across PDF viewers without requiring locally installed fonts or an office application.

## Historical execution versus analysis reproduction

The `experiment/` directory contains the available source snapshot for inspection. Some implementation paths were revised after the spring runs, and the historical dirty working tree was not fully captured. Endpoint aliases also do not freeze provider internals. Consequently, this release promises reproduction of the saved-data analysis, not identical regenerated model answers. Historical configuration values must not be treated as current service specifications.

Supplementary observations are kept separately in `data/supplementary/`. Their role and selection are described in [provenance](provenance.md). They are not merged into the main experiment or used to tune the final historical scores.
