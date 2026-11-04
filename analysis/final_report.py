"""Build the effective-context report from saved observations; no model calls.

This module deliberately does not fit or extrapolate a context-capacity curve.
The frozen November analysis supplies scoring and same-input method contrasts.
Additional analyses describe token workload and task-matched length conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.research_report import (
    METHODS, LABELS, analysis_population, build, wilson,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["NIAH", "BABILong_qa2", "Banking77_ICL"]
TASK_LABELS = {"NIAH": "Single-fact retrieval", "BABILong_qa2": "Multi-hop reasoning",
               "Banking77_ICL": "Intent classification"}
COLORS = ["#213d53", "#287d8e", "#b06c39", "#7c599b"]
CONFIGS = ["16k", "32k", "64k"]


def load_observations(path: Path, manifest: Path) -> pd.DataFrame:
    expected = json.loads(manifest.read_text(encoding="utf-8"))["sha256"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Observation checksum differs from its manifest")
    frame = pd.read_csv(path, dtype={"gold_answer": str, "source_task_id": str,
                                    "cluster_id": str}, keep_default_na=False)
    flags = frame.provider_fail.astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Unrecognized provider-failure flag")
    frame["provider_fail"] = flags.eq("true")
    return frame


def collapse_source_conditions(retained: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Average repeated identical inputs, never select the better answer."""
    frame = retained[retained.dataset_t.eq("BABILong_qa2")].copy()
    frame["source_config"] = frame.input_text_hash.map(lambda h: mapping[h]["config"])
    for row in frame.itertuples():
        source = mapping[row.input_text_hash]
        if (str(source["row_index"]) != str(row.source_task_id)
                or f"babi-prefix-{source['prefix_family_id']}" != row.cluster_id):
            raise ValueError("Observation and recovered source identities disagree")
    keys = ["procedure_A", "model", "source_task_id", "cluster_id", "source_config"]
    grouped = frame.groupby(keys, sort=True)
    identity = grouped[["input_text_hash", "gold_answer", "actual_words"]].nunique()
    if (identity != 1).any().any():
        raise ValueError("A source-condition group contains different inputs or answers")
    return grouped.agg(score=("metric_value", "mean"), repeats=("metric_value", "size"),
                       actual_words=("actual_words", "first"),
                       input_text_hash=("input_text_hash", "first")).reset_index()


def bootstrap_clusters(pair: pd.DataFrame, repeats: int = 20000) -> tuple[float, float]:
    # Lexicographic group order is part of the reproducibility specification.
    groups = pair.groupby("cluster_id", sort=True).delta.agg(["sum", "size"])
    rng = np.random.default_rng(20261102)
    indices = rng.integers(0, len(groups), (repeats, len(groups)))
    draws = (groups["sum"].to_numpy()[indices].sum(axis=1)
             / groups["size"].to_numpy()[indices].sum(axis=1))
    return tuple(np.quantile(draws, [.025, .975]))


def paired_source_lengths(collapsed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, units = [], []
    key = ["model", "source_task_id", "cluster_id"]
    for method in METHODS:
        subset = collapsed[collapsed.procedure_A.eq(method)]
        for first, second in [("16k", "32k"), ("16k", "64k"), ("32k", "64k")]:
            pair = subset[subset.source_config.eq(first)].merge(
                subset[subset.source_config.eq(second)], on=key,
                suffixes=("_short", "_long"), validate="one_to_one")
            if pair.empty:
                continue
            pair["delta"] = pair.score_long-pair.score_short
            low, high = bootstrap_clusters(pair)
            label = f"{first} to {second}"
            results.append({"procedure": method, "source_comparison": label,
                "task_ids": pair.source_task_id.nunique(), "prefix_families": pair.cluster_id.nunique(),
                "model_task_pairs": len(pair), "difference": pair.delta.mean(),
                "ci_low": low, "ci_high": high,
                "mean_short_words": pair.actual_words_short.mean(),
                "mean_long_words": pair.actual_words_long.mean()})
            pair["procedure"] = method
            pair["source_comparison"] = label
            units.append(pair)
    return pd.DataFrame(results), pd.concat(units, ignore_index=True)


def grid_pattern(accuracy, threshold=.5) -> str:
    """Describe observed means, not population capacity or significance."""
    passed = np.asarray(accuracy, dtype=float) >= threshold
    if not len(passed):
        raise ValueError("An observed profile cannot be empty")
    if passed.all():
        return "all_observed_means_at_or_above_threshold"
    if not passed.any():
        return "all_observed_means_below_threshold"
    if np.any(np.diff(passed.astype(int)) > 0):
        return "later_recovery_or_nonmonotone_threshold_pattern"
    return "observed_mean_crossing_without_later_recovery"


def make_figures(cells: pd.DataFrame, comparisons: pd.DataFrame, paired: pd.DataFrame,
                 retained: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#243b4b", "text.color": "#243b4b",
                         "axes.titleweight": "semibold", "svg.hashsalt": "effective-context-2026"})
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ["png", "svg"]:
            fig.savefig(out/f"{name}.{ext}", dpi=190, bbox_inches="tight",
                        metadata={"Date": None} if ext == "svg" else {})
        plt.close(fig)

    # A worked example selected for coverage of all three tasks, not best score.
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex="col")
    for col, dataset in enumerate(DATASETS):
        for method, color in zip(METHODS, COLORS):
            view = cells[(cells.dataset_t == dataset)&(cells.model == "gpt-4o-mini")
                         &(cells.procedure_A == method)].sort_values("mean_actual_words")
            x = view.mean_actual_words/1000
            axes[0, col].errorbar(x, view.accuracy,
                yerr=[view.accuracy-view.ci_low, view.ci_high-view.accuracy],
                fmt="o", markersize=4, capsize=2, color=color, alpha=.85, label=LABELS[method])
            axes[1, col].plot(x, view.mean_input_tokens/1000, "o", color=color, markersize=5)
        axes[0, col].set_title(TASK_LABELS[dataset], fontsize=12.5)
        axes[0, col].set_ylim(-.04, 1.05)
        axes[0, col].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Source length\n(thousands of words)", fontsize=11)
        for row in range(2):
            axes[row, col].grid(alpha=.16)
    axes[0, 0].set_ylabel("Observed accuracy")
    axes[1, 0].set_ylabel("Summed input tokens\n(thousands; log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=4, frameon=False)
    fig.suptitle("GPT-4o-mini: quality and model input across tasks", y=.998, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, "context_profiles")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, dataset in zip(axes, DATASETS[1:]):
        view = comparisons[comparisons.dataset.eq(dataset)].set_index("procedure")
        for y, (method, color) in enumerate(zip(METHODS[1:], COLORS[1:])):
            row = view.loc[method]
            ax.errorbar(100*row.difference, y,
                xerr=[[100*(row.difference-row.ci_low)], [100*(row.ci_high-row.difference)]],
                fmt="o", color=color, markersize=7, capsize=4)
        ax.axvline(0, color="#a1a9af", linewidth=1)
        ax.set_yticks(range(3), [LABELS[m] for m in METHODS[1:]])
        ax.set_title(TASK_LABELS[dataset], fontsize=12)
        ax.set_xlabel("Difference from full context\n(percentage points)", fontsize=11)
        ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    save(fig, "method_differences")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), sharey=True)
    models = sorted(retained.model.unique())
    for ax, dataset in zip(axes, DATASETS):
        table = retained[retained.dataset_t.eq(dataset)].groupby(["model", "procedure_A"]).metric_value.mean().unstack()
        table = table.reindex(index=models, columns=METHODS)
        matrix = table.to_numpy(dtype=float)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#eef0f2")
        ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for i in range(len(models)):
            for j in range(len(METHODS)):
                value = matrix[i, j]
                ax.text(j, i, "--" if np.isnan(value) else f"{100*value:.0f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if value > .65 else "#243b4b")
        ax.set_xticks(range(4), ["Full", "RAG", "ReAct", "Memory"], rotation=35, ha="right")
        ax.set_yticks(range(len(models)), models, fontsize=10.5)
        ax.set_title(dataset.replace("_ICL", "").replace("_qa2", " qa2"), fontsize=12)
    fig.suptitle("Observed accuracy (%) across the available model-task combinations", fontsize=13)
    fig.tight_layout()
    save(fig, "model_task_matrix")

    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    view = paired[paired.procedure.eq("A0_baseline")]
    for y, row in enumerate(view.itertuples()):
        ax.errorbar(100*row.difference, y,
            xerr=[[100*(row.difference-row.ci_low)], [100*(row.ci_high-row.difference)]],
            fmt="o", capsize=4, markersize=7, color=COLORS[0])
        ax.text(15, y, f"{row.task_ids} shared tasks", va="center", fontsize=9)
    ax.axvline(0, color="#a1a9af", linewidth=1)
    ax.set_yticks(range(len(view)), view.source_comparison)
    ax.set_xlabel("Longer-source minus shorter-source accuracy (percentage points)")
    ax.set_title("Task-matched comparisons: full-context baseline", fontsize=12)
    ax.set_xlim(-27, 25)
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    save(fig, "matched_lengths")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/historical/observations.csv.gz")
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/historical/manifest.json")
    parser.add_argument("--mapping", type=Path, default=ROOT/"data/historical/babilong_source_ids.json")
    parser.add_argument("--workload", type=Path, default=ROOT/"data/historical/workload_evidence.csv")
    parser.add_argument("--out", type=Path, default=ROOT/"results/final")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    data = load_observations(args.data, args.manifest)
    info = build(data, args.out, make_figures=False)
    _, retained = analysis_population(data)
    evidence = pd.read_csv(args.workload)
    if len(evidence) != 40:
        raise ValueError("Expected the 40-record worked-example evidence")
    for row in evidence.itertuples():
        ins, outs = json.loads(row.call_input_tokens), json.loads(row.call_output_tokens)
        if sum(ins) != row.input_tokens or sum(outs) != row.output_tokens or len(ins) != row.model_calls:
            raise ValueError("Workload trace does not match episode usage")
        match = retained[(retained.input_text_hash == row.input_text_hash)
                         & (retained.model == row.model) & (retained.procedure_A == row.procedure_A)
                         & (retained.target_words == row.target_words)]
        if len(match) != 1 or match.input_tokens.iloc[0] != row.input_tokens or match.output_tokens.iloc[0] != row.output_tokens:
            raise ValueError("Workload evidence disagrees with observation export")
    evidence.groupby("procedure_A").agg(n=("model_calls", "size"),
        mean_input_tokens=("input_tokens", "mean"), mean_output_tokens=("output_tokens", "mean"),
        mean_model_calls=("model_calls", "mean")).reset_index().to_csv(args.out/"workload_example.csv", index=False)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))["items"]
    collapsed = collapse_source_conditions(retained, mapping)
    paired, pairs = paired_source_lengths(collapsed)
    collapsed.to_csv(args.out/"babilong_collapsed_conditions.csv", index=False)
    paired.to_csv(args.out/"matched_length_differences.csv", index=False)
    pairs.to_csv(args.out/"matched_length_pairs.csv", index=False)

    group = ["dataset_t", "model", "procedure_A", "target_words"]
    cells = retained.groupby(group, sort=True).agg(n=("metric_value", "size"),
        correct=("metric_value", "sum"), accuracy=("metric_value", "mean"),
        mean_actual_words=("actual_words", "mean"), min_actual_words=("actual_words", "min"),
        max_actual_words=("actual_words", "max"), mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"), unique_clusters=("cluster_id", "nunique")).reset_index()
    cells["ci_low"], cells["ci_high"] = wilson(cells.correct, cells.n)
    cells.to_csv(args.out/"context_profiles.csv", index=False)

    patterns = []
    for keys, view in cells.groupby(group[:3], sort=True):
        view = view.sort_values("mean_actual_words")
        patterns.append(dict(zip(group[:3], keys)) | {
            "threshold": .5, "interpretation": "descriptive observed means only; not a capacity estimate",
            "pattern": grid_pattern(view.accuracy), "conditions": len(view)})
    pd.DataFrame(patterns).to_csv(args.out/"observed_threshold_patterns.csv", index=False)

    common = []
    for method, view in collapsed.groupby("procedure_A", sort=True):
        counts = view.groupby(["model", "source_task_id"]).source_config.nunique()
        ids = counts[counts.eq(3)].reset_index()[["model", "source_task_id"]]
        shared = view.merge(ids, on=["model", "source_task_id"], validate="many_to_one")
        for config, part in shared.groupby("source_config", sort=True):
            common.append({"procedure": method, "source_config": config,
                "tasks": part.source_task_id.nunique(), "prefix_families": part.cluster_id.nunique(),
                "model_task_pairs": len(part), "accuracy": part.score.mean()})
    pd.DataFrame(common).to_csv(args.out/"common_task_panel.csv", index=False)
    model_task = retained.groupby(["dataset_t", "model", "procedure_A"], sort=True).agg(
        n=("metric_value", "size"), correct=("metric_value", "sum"), accuracy=("metric_value", "mean")).reset_index()
    model_task.to_csv(args.out/"model_task_accuracy.csv", index=False)
    info.update({"report_scope": "Effective context across tasks and processing methods",
        "capacity_estimates": "None. Observed profiles and threshold-pattern descriptions only.",
        "observation_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "source_mapping_sha256": hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
        "paired_lengths": "Same source task, different source configurations/backgrounds; not isolated length effects. Repeated identical inputs averaged; fixed models; lexicographically ordered prefix clusters; 20000 draws; seed 20261102.",
        "threshold_pattern": "0.5 is a descriptive legacy cutoff, not a validated utility requirement or a confidence bound."})
    (args.out/"analysis_manifest.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8")
    if not args.no_figures:
        make_figures(cells, pd.read_csv(args.out/"paired_cluster_differences.csv"), paired,
                     retained, args.out/"figures")
    print(json.dumps({"observations": len(data), "retained": len(retained),
                      "output": str(args.out), "model_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
