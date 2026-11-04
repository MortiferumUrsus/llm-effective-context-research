"""Reproduce descriptive historical results without fitting context boundaries.

Input can be the original local manifest or a reviewed observation CSV. The
export contains an explicit allowlist of fields, never provider error bodies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

METHODS = ["A0_baseline", "A_rag", "A_ReAct", "A_MemGPT"]
LABELS = {"A0_baseline": "Full context", "A_rag": "Hybrid RAG",
          "A_ReAct": "ReAct-style", "A_MemGPT": "Memory-style"}
KEY = ["dataset_t", "model", "target_words", "seed", "episode_idx"]
EXPORT_COLUMNS = KEY + ["procedure_A", "input_text_hash", "actual_words",
    "input_tokens", "output_tokens", "wall_clock_ms", "metric_value", "metric_name",
    "finish_reason", "provider_fail", "gold_answer", "output", "timestamp",
    "git_commit", "test_idx", "cluster_id", "source_task_id", "gold_provenance"]


def analysis_population(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    if data.duplicated(KEY+["procedure_A"]).any():
        raise ValueError("Duplicate canonical observation keys")
    if not data.metric_value.isin([0, 1]).all():
        raise ValueError("Expected binary recorded scores")
    bad_provider = data.provider_fail.fillna(False).astype(bool)
    overflow = data.finish_reason.eq("ctx_overflow")
    data["retained"] = ~(bad_provider | overflow)
    data["availability_reason"] = np.select([bad_provider, overflow],
                                             ["provider_failure", "context_overflow"], "retained")
    data["correct_per_attempt"] = data.metric_value.where(data.retained, 0)
    return data, data.loc[data.retained].copy()


def wilson(correct, total, alpha=.05):
    z = norm.ppf(1-alpha/2)
    p = np.asarray(correct, dtype=float)/np.asarray(total, dtype=float)
    n = np.asarray(total, dtype=float)
    den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return np.maximum(0, mid-half), np.minimum(1, mid+half)


def strict_answer(output: str, gold: str, dataset: str) -> bool:
    """Declared conservative sensitivity rule; historical scores remain primary.

    Accept one final answer line, optionally prefixed by Answer/Label, quotes,
    an article for locations, and terminal punctuation. Explanations and
    multiple candidate answers are rejected, not adjudicated opportunistically.
    """
    text = str(output).strip().lower()
    if not str(gold).strip() or pd.isna(gold):
        raise ValueError("Answer sensitivity requires a verified gold answer")
    if dataset == "NIAH":
        numbers = set(re.findall(r"(?<!\d)\d{7}(?!\d)", text))
        return numbers == {str(gold).strip()}
    marked = re.findall(r"(?:^|\n)\s*(?:answer|label)\s*:\s*([^\n]+)", text)
    if len(marked) > 1:
        return False
    if marked:
        last_line = [line.strip() for line in text.splitlines() if line.strip()][-1]
        if not re.match(r"^(?:answer|label)\s*:", last_line):
            return False
        text = marked[0]
    text = text.strip(" \t\r\n\"'`.*")
    if dataset == "BABILong_qa2":
        text = re.sub(r"^the\s+", "", text)
    return text == str(gold).strip().lower()


def matched(data: pd.DataFrame, method: str) -> pd.DataFrame:
    pair = data[data.procedure_A == "A0_baseline"].merge(
        data[data.procedure_A == method], on=KEY, suffixes=("_a", "_b"),
        validate="one_to_one")
    for column in ["input_text_hash", "gold_answer", "cluster_id"]:
        if not pair[column+"_a"].eq(pair[column+"_b"]).all():
            raise ValueError(f"Paired observations disagree on {column}")
    return pair


def clustered_difference(data: pd.DataFrame, method: str, repeats=20000) -> dict:
    pair = matched(data, method)
    if pair.empty:
        return {}
    differences = pair.metric_value_b-pair.metric_value_a
    groups = pd.DataFrame({"id": pair.cluster_id_a, "delta": differences}).groupby("id").delta.agg(["sum", "size"])
    def cluster_order(identifier):
        value = str(identifier)
        match = re.search(r"\d+$", value)
        return (value[:match.start()], int(match.group())) if match else (value, 0)
    groups = groups.reindex(sorted(groups.index, key=cluster_order))
    rng = np.random.default_rng(20261102)
    indices = rng.integers(0, len(groups), (repeats, len(groups)))
    draws = groups["sum"].to_numpy()[indices].sum(axis=1)/groups["size"].to_numpy()[indices].sum(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return {"dataset": data.dataset_t.iloc[0], "procedure": method,
            "pairs": len(pair), "clusters": len(groups), "difference": differences.mean(),
            "ci_low": low, "ci_high": high}


def attach_clusters(data: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["items"]
    result = data.copy()
    result["cluster_id"] = ""
    result["source_task_id"] = ""
    result["gold_provenance"] = "recorded"
    niah_recovered = {}
    for index, row in result.iterrows():
        if row.dataset_t == "BABILong_qa2":
            source = mapping[row.input_text_hash]
            cluster, task = f"babi-prefix-{source['prefix_family_id']}", source["row_index"]
        elif row.dataset_t == "Banking77_ICL":
            if pd.isna(row.test_idx):
                raise ValueError("Missing Banking77 held-out query ID")
            cluster = f"banking-query-{int(row.test_idx)}"
            task = int(row.test_idx)
        elif row.dataset_t == "NIAH":
            # Early NIAH records lack gold metadata. Recover it only after an
            # exact input-hash match, never from the model's output or score.
            identity = (int(row.seed), int(row.episode_idx), int(row.target_words))
            if identity not in niah_recovered:
                from bench.niah import make_episode
                episode = make_episode(*identity)
                niah_recovered[identity] = (episode.haystack_hash, episode.value)
            expected_hash, answer = niah_recovered[identity]
            if row.input_text_hash != expected_hash:
                raise ValueError("NIAH generator does not reproduce the historical input hash")
            if pd.notna(row.gold_answer) and str(row.gold_answer).strip() and str(row.gold_answer) != answer:
                raise ValueError("NIAH recovered answer disagrees with recorded gold")
            result.at[index, "gold_provenance"] = (
                "recorded_and_input_hash_verified" if pd.notna(row.gold_answer) and str(row.gold_answer).strip()
                else "recovered_from_hash_matched_generator")
            result.at[index, "gold_answer"] = answer
            cluster, task = f"niah-answer-{answer}", answer
        else:
            raise ValueError(f"No declared independent-unit policy for {row.dataset_t}")
        result.at[index, "cluster_id"] = cluster
        result.at[index, "source_task_id"] = str(task)
    return result


def export_observations(data: pd.DataFrame, path: Path):
    export = data.reindex(columns=EXPORT_COLUMNS).copy()
    # Fail closed if response fields unexpectedly contain a credential or an
    # account-management URL; do not silently publish a partly scrubbed record.
    sensitive = re.compile(r"sk-or-v1-[a-zA-Z0-9]+|Bearer\s+[A-Za-z0-9._-]+|openrouter\.ai/[^\s]*?(?:keys|workspaces)", re.I)
    if export.output.fillna("").astype(str).str.contains(sensitive).any():
        raise ValueError("Observation outputs require manual redaction before export")
    path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def figures(cells: pd.DataFrame, comparisons: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#475569", "#2563eb", "#d97706", "#9333ea"]
    out.mkdir(parents=True, exist_ok=True)
    for dataset in ["BABILong_qa2", "Banking77_ICL"]:
        subset = cells[cells.dataset_t == dataset]
        models = sorted(subset.model.unique())
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, squeeze=False)
        for ax, model in zip(axes.flat, models):
            for method, color in zip(METHODS, colors):
                values = subset[(subset.model == model)&(subset.procedure_A == method)].sort_values("mean_actual_words")
                # Points and nominal Wilson intervals, not fitted curves or
                # inferred continuous capacity boundaries.
                ax.errorbar(values.mean_actual_words/1000, values.accuracy,
                    yerr=[values.accuracy-values.ci_low, values.ci_high-values.accuracy],
                    fmt="o", ms=3, capsize=2, alpha=.8, color=color, label=LABELS[method])
            ax.set_title(model, fontsize=10)
            ax.set_ylim(-.03, 1.03)
            ax.grid(alpha=.15)
            ax.set_xlabel("Mean source length (thousands of words)", fontsize=8)
        axes[0, 0].set_ylabel("Recorded accuracy")
        axes[1, 0].set_ylabel("Recorded accuracy")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        fig.suptitle(f"{dataset}: historical observed conditions", y=.96, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .92))
        for ext in ["png", "svg"]:
            fig.savefig(out/f"{dataset.lower()}_observed.{ext}", dpi=170)
        plt.close(fig)
    selected = comparisons[comparisons.dataset.isin(["BABILong_qa2", "Banking77_ICL"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, dataset in zip(axes, ["BABILong_qa2", "Banking77_ICL"]):
        view = selected[selected.dataset == dataset].set_index("procedure").reindex(METHODS[1:])
        for y, (method, row) in enumerate(view.iterrows()):
            ax.errorbar(100*row.difference, y,
                        xerr=[[100*(row.difference-row.ci_low)], [100*(row.ci_high-row.difference)]],
                        fmt="o", color=colors[y+1], capsize=4)
        ax.axvline(0, color="#64748b", lw=1)
        ax.set_yticks(range(3), [LABELS[m] for m in METHODS[1:]])
        ax.set_title(dataset)
        ax.set_xlabel("Accuracy difference from full context (percentage points)", fontsize=8)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Exploratory paired differences with task-cluster sensitivity intervals", fontsize=11)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(out/f"paired_differences.{ext}", dpi=170)
    plt.close(fig)


def build(data: pd.DataFrame, out: Path, make_figures=True) -> dict:
    attempted, retained = analysis_population(data)
    retained["strict_sensitivity_score"] = [strict_answer(r.output, r.gold_answer, r.dataset_t)
                                             for r in retained.itertuples()]
    group = ["dataset_t", "procedure_A"]
    summary = attempted.groupby(group).agg(attempted=("metric_value", "size"),
        retained=("retained", "sum"), correct=("correct_per_attempt", "sum")).reset_index()
    summary["availability"] = summary.retained/summary.attempted
    summary["conditional_accuracy"] = summary.correct/summary.retained
    summary["correct_per_attempt"] = summary.correct/summary.attempted
    cells = retained.groupby(["dataset_t", "model", "procedure_A", "target_words"]).agg(
        n=("metric_value", "size"), correct=("metric_value", "sum"), accuracy=("metric_value", "mean"),
        mean_actual_words=("actual_words", "mean"), min_actual_words=("actual_words", "min"),
        max_actual_words=("actual_words", "max"), mean_input_tokens=("input_tokens", "mean"),
        unique_clusters=("cluster_id", "nunique")).reset_index()
    cells["ci_low"], cells["ci_high"] = wilson(cells.correct, cells.n)
    contrasts = []
    for _, values in retained.groupby("dataset_t"):
        for method in METHODS[1:]:
            contrast = clustered_difference(values, method)
            if contrast:
                contrasts.append(contrast)
    comparisons = pd.DataFrame(contrasts)
    sensitivity = retained.groupby(group).agg(recorded_accuracy=("metric_value", "mean"),
        strict_accuracy=("strict_sensitivity_score", "mean"), n=("metric_value", "size")).reset_index()
    out.mkdir(parents=True, exist_ok=True)
    for name, table in [("availability_and_accuracy", summary), ("observed_cells", cells),
                        ("paired_cluster_differences", comparisons), ("answer_sensitivity", sensitivity)]:
        table.to_csv(out/f"{name}.csv", index=False)
    info = {"attempted": len(attempted), "retained": len(retained),
            "excluded": int((~attempted.retained).sum()),
            "dates": [str(data.timestamp.min()), str(data.timestamp.max())],
            "analysis_date": "2026-11-02", "original_study_period": "Spring 2026",
            "inference": "Historical exploratory observations; fixed model set. BABILong clusters are exact-prefix families, not guaranteed independent source episodes.",
            "cell_intervals": "Nominal Wilson intervals assuming independent Bernoulli observations within each cell; story dependence limits this interpretation.",
            "cluster_intervals": "20000 percentile draws, seed 20261102; sensitivity analysis, not simultaneous confidence bands. Degenerate intervals on saturated NIAH are uninformative and should not imply exact certainty."}
    (out/"analysis_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if make_figures:
        figures(cells, comparisons, out/"figures")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest")
    inputs.add_argument("--observations", type=Path)
    parser.add_argument("--mapping", type=Path,
                        default=Path("audit/2026-11-02/evidence/babilong_source_ids.json"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/revision_2026_11"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.manifest:
        from analysis.extract_metrics import load_from_manifest
        data = attach_clusters(load_from_manifest(args.manifest, strict=True), args.mapping)
    else:
        data = pd.read_csv(args.observations, dtype={"gold_answer": str, "cluster_id": str,
                                                   "source_task_id": str}, keep_default_na=False)
        data["provider_fail"] = data.provider_fail.map(lambda value: str(value).lower() == "true")
    if args.export:
        export_observations(data, args.export)
    print(json.dumps(build(data, args.out, not args.no_figures), indent=2))


if __name__ == "__main__":
    main()
