import logging
import os
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd

from dgm_eval.utils import get_k_substring, get_metric_substring

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)

METRIC_COLS = {
    "prdc": ["precision", "recall", "prdc_nreal", "prdc_nfake"],
    "knn-filter": [
        "knn-filter-p",
        "knn-filter-r",
        "knn-filter-nreal",
        "knn-filter-nfake",
    ],
}

METRIC_RENAME = {
    "precision": "P",
    "recall": "R",
    "density": "D",
    "coverage": "C",
    "prdc_nreal": "N",
    "prdc_nfake": "M",
    "knn-filter-p": "P (knn-filter)",
    "knn-filter-r": "R (knn-filter)",
    "knn-filter-nreal": "N",
    "knn-filter-nfake": "M",
}

PRECISION_EST_STR = r"$\hat \alpha (P_X, Q_X)$"
COND_PRECISION_EST_STR = lambda i: (
    rf"$\hat \alpha (P_{{X|Y={{{i}}}}}, Q_{{X|Y={{{i}}}}})$"
)
RAND_COND_PRECISION_EST_STR = lambda i: (
    rf"$\hat \alpha (P_{{X|\cdot={{{i}}}}}, Q_{{X|\cdot={{{i}}}}})$"
)
AVG_COND_PRECISION_EST_STR = r"$E_Y \left[ \hat \alpha(P_{X|Y}, Q_{X|Y}) \right]$"
AVG_RAND_COND_PRECISION_EST_STR = r"$E_Z \left[ \hat \alpha(P_{X|Z}, Q_{X|Z}) \right]$"
DIFF_STR = r"$\Delta$"


def fmt_cell(row, col, df_mean, df_std):
    mean = df_mean.loc[row, col]
    std = df_std.loc[row, col] if col in df_std.columns else np.nan
    if pd.isna(std) or std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} $\pm$ {std:.3f}"


def fmt_block(df_mean, df_std, metric_order, body_rows):
    """Format one source into "mean $\\pm$ std" cells for the body rows.

    Parameters
    ----------
    df_mean : pd.DataFrame
        DataFrame with mean values
    df_std : pd.DataFrame
        DataFrame with std values
    metric_order : list
        Column names to include in order
    body_rows : list
        Row names to include

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with cells as strings
    """
    return pd.DataFrame(
        {
            col: [fmt_cell(r, col, df_mean, df_std) for r in body_rows]
            for col in metric_order
        },
        index=body_rows,
    )


def select_best_worst_indexes(scores, n_each=5):
    """Indices of the ``n_each`` highest- and ``n_each`` lowest-scoring labels.

    Parameters
    ----------
    scores : array-like
        Per-label metric, ``scores[i]`` is the score of label ``i``.
    n_each : int
        Number of best and of worst labels to keep.

    The selection is made on the score, but the returned indices are sorted by
    index (not by score). If there are at most ``2 * n_each`` labels, all indices
    are returned.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n <= 2 * n_each:
        return list(range(n))
    order = np.argsort(scores)  # ascending by score
    selection = set(order[:n_each].tolist()) | set(order[-n_each:].tolist())
    return sorted(selection)


def load_mean_std(path):
    """Load a result file and split it into mean / std DataFrames.

    Scores carry per-label rows plus their "*_std" counterparts (variance
    across runs). For randomized labels that variance is over the random
    label draws; for true labels it is over the prdc subsampling.

    Returns raw DataFrames WITHOUT column renaming - caller applies METRIC_RENAME at the end.
    """
    data = np.load(path, allow_pickle=True)
    df = pd.DataFrame(data["scores"].item()["run00"]).T

    mean_rows = [r for r in df.index if not r.endswith("_std")]
    std_rows = [r for r in df.index if r.endswith("_std")]

    df_mean = df.loc[mean_rows].copy()
    df_std = df.loc[std_rows].rename(index=lambda x: x.removesuffix("_std")).copy()
    return df_mean, df_std


def save_latex_table(
    df: pd.DataFrame,
    outdir: str | Path,
    fname: str,
    format_kwargs=None,
    to_latex_kwargs=None,
    topleft=None,
):
    """
    Save a DataFrame as a LaTeX table.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    outdir : str | Path
        Directory where the LaTeX file will be saved.
    fname : str
        Name of the LaTeX file (without extension).
    topleft : str, optional
        String to display in the top-left corner.
    """
    logger.info("Generating LaTeX table for DataFrame:")
    print(df)

    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    out = outdir / f"{fname}.tex"

    styler = df.style.format(
        decimal=".",
        thousands=",",
        precision=3,
        **(format_kwargs or {}),
    )
    latex = styler.to_latex(
        hrules=True,
        multicol_align="c",
        **(to_latex_kwargs or {}),
    )

    # Add metric name to top-left corner if provided
    if topleft is not None:
        print(f"TOPLEFT: '{topleft}'")
        # Replace the first empty cell in the header row (before the first &)
        latex = re.sub(
            r"(\\begin\{tabular\}.*?\n)([ \t]*&)",
            rf"\1{topleft} &",
            latex,
            count=1,
            flags=re.DOTALL,
        )

    out.write_text(latex, encoding="utf-8")
    logger.info(f"Saved LaTeX table to {out.resolve()}")


####################################################################################################
####################################################################################################
####################################################################################################


@click.group()
def main():
    """
    This scipt processes results and generates LaTeX tables.
    """


####################################################################################################
# DESCRIPTIVE STATISTICS
####################################################################################################


# @main.command()
# @click.option("--train",            help="Paths to the images: real dataset.",                          type=str)  # fmt: skip
# @click.option("--gen",              help="Paths to the images, generated dataset.",                     type=str, nargs="+")  # fmt: skip
# @click.option("--outdir", "-o",     help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="out-latexify")  # fmt: skip
# def descriptive_statistics(train, gen, outdir):

#     real_dl = get_dataloader_from_path(
#         train,
#         model.transform,
#         num_workers,
#         args,
#     )  # list


####################################################################################################
# XP STANDARD
####################################################################################################


@main.command()
@click.option("--path", "-p",       help="Path containing the sweep results",                           type=click.Path(exists=True))  # fmt: skip
@click.option("--outdir", "-o",     help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="out-latexify")  # fmt: skip
def xp(path, outdir):
    logger.info(f"Processing results in {path}...")

    # Output fname
    fname = "xp_" + Path(path).stem
    logger.info(f"Derived LaTeX table name: {fname}")

    # Data
    data = np.load(path, allow_pickle=True)
    metrics = data["scores"].item()["run00"]
    df = pd.DataFrame(metrics).T

    # Split mean and std rows
    mean_rows = [r for r in df.index if not r.endswith("_std")]
    std_rows = [r for r in df.index if r.endswith("_std")]
    num_labels = sum(1 for r in mean_rows if r.startswith("label-"))

    df_mean = df.loc[mean_rows].copy()
    df_std = df.loc[std_rows].rename(index=lambda x: x.removesuffix("_std")).copy()

    metric_cols = [
        "precision",
        # "recall",
        # "density",
        # "coverage",
    ]
    count_col = "prdc_nreal"  # prdc_nreal == prdc_nfake always

    # diff row: overall - agg, no std
    df_mean.loc["diff", metric_cols] = (
        df_mean.loc["overall", metric_cols] - df_mean.loc["agg", metric_cols]
    )
    df_mean.loc["diff", count_col] = np.nan
    df_std.loc["diff"] = np.nan

    # format cells
    col_keys = [count_col] + metric_cols
    df_display = pd.DataFrame(
        {
            col: [fmt_cell(row, col, df_mean, df_std) for row in df_mean.index]
            for col in col_keys
        },
        index=df_mean.index,
    )

    # Display the 5 best and 5 worst labels by precision (ordered by index)
    label_scores = [df_mean.loc[f"label-{i}", "precision"] for i in range(num_labels)]
    selected_labels = select_best_worst_indexes(label_scores, n_each=5)
    label_rows = [f"label-{i}" for i in selected_labels]
    row_keys = ["overall"] + label_rows + ["agg", "diff"]

    row_rename = {
        "overall": PRECISION_EST_STR,
        "agg": AVG_COND_PRECISION_EST_STR,
        "diff": DIFF_STR,
        **{f"label-{i}": COND_PRECISION_EST_STR(i) for i in selected_labels},
    }
    col_rename = {
        count_col: "N",
        "precision": "P",
        "recall": "R",
        "density": "D",
        "coverage": "C",
    }

    df_final = df_display.loc[row_keys, col_keys].rename(
        index=row_rename, columns=col_rename
    )

    if outdir is not None:
        outdir = Path(outdir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        save_latex_table(df_final, outdir=outdir, fname=fname)

    return df_final, fname


####################################################################################################
# METRIC KNN-FILTER
####################################################################################################


@main.command()
@click.option("--dir", "-d",        help="Directory containing the knn-filter metric results",                      type=click.Path(exists=True))  # fmt: skip
@click.option("--outdir", "-o",     help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="out-latexify")  # fmt: skip
def metric_knn_filter(dir, outdir):
    logger.info(f"Processing knn-filter metric results in {dir}...")

    paths = []
    for fname in os.listdir(dir):
        if fname.endswith(".npz"):
            path = os.path.join(dir, fname)
            paths.append(path)
            logger.info(f"Found result file: {path}")
    paths.sort()

    fname = "metric-knn-filter_" + Path(paths[0]).stem.split("_k-")[0]
    logger.info(f"Derived LaTeX table name: {fname}")

    metric = "P"  # report precision only

    records = []  # one row per k: overall / avg / diff precision
    for path in paths:
        k_val = Path(path).stem.split("-")[-1]  # "1", "2", "None"
        k_sort = pd.to_numeric(k_val, errors="coerce")
        k_val = r"$\sqrt{n}$" if k_val == "None" else k_val
        logger.info(f"Fetching results for k = {k_val}")

        data = np.load(path, allow_pickle=True)
        run = data["scores"].item()["run00"]

        # knn-balls-filtering nests every per-label result under "overall"; the
        # across-run std (if any) mirrors that structure under "overall_std".
        overall = run["overall"]
        overall_std = run.get("overall_std", {})
        label_keys = [k for k in overall if k.startswith("label-")]

        df_mean = pd.DataFrame(
            index=["overall", "avg", "diff"], columns=[metric], dtype=float
        )
        df_std = pd.DataFrame(
            index=["overall", "avg", "diff"], columns=[metric], dtype=float
        )

        # overall: precision of the marginal distributions (has an across-run std)
        df_mean.loc["overall", metric] = overall["precision"]
        df_std.loc["overall", metric] = (
            overall_std.get("precision", np.nan) if overall_std else np.nan
        )

        # avg: mean over the per-label precisions (mean only; the std of the
        # mean-over-labels is not recoverable from the per-label stds)
        df_mean.loc["avg", metric] = np.mean(
            [overall[k]["precision"] for k in label_keys]
        )
        df_std.loc["avg", metric] = np.nan

        # diff: overall - avg (mean only)
        df_mean.loc["diff", metric] = (
            df_mean.loc["overall", metric] - df_mean.loc["avg", metric]
        )
        df_std.loc["diff", metric] = np.nan

        records.append(
            {
                "k": k_val,
                "k_sort": k_sort,
                "overall": fmt_cell("overall", metric, df_mean, df_std),
                "avg": fmt_cell("avg", metric, df_mean, df_std),
                "diff": fmt_cell("diff", metric, df_mean, df_std),
            }
        )

    df_final = (
        pd.DataFrame(records)
        .sort_values("k_sort")
        .drop(columns="k_sort")
        .set_index("k")
        .rename(
            columns={
                "overall": PRECISION_EST_STR,
                "avg": AVG_COND_PRECISION_EST_STR,
                "diff": DIFF_STR,
            }
        )
    )
    df_final.index.name = "k"

    format_kwargs = {"na_rep": ""}
    to_latex_kwargs = {"column_format": "c" * (len(df_final.columns) + 1)}

    if outdir is not None:
        outdir = Path(outdir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        save_latex_table(
            df_final,
            outdir=outdir,
            fname=fname,
            format_kwargs=format_kwargs,
            to_latex_kwargs=to_latex_kwargs,
        )

    return df_final, fname


####################################################################################################
# XP SWEEP PRDC K
####################################################################################################


@main.command()
@click.option("--dir", "-d",        help="Directory containing the sweep results",                      type=click.Path(exists=True))  # fmt: skip
@click.option("--metric", "-m",     help="Metric name to extract (e.g., 'prdc', 'knn-filter')",         type=str, required=True)  # fmt: skip
@click.option("--outdir", "-o",     help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="out-latexify")  # fmt: skip
def xp_sweep_prdc_k(dir, metric, outdir):
    logger.info(f"\nProcessing sweep results in {dir} for metric: {metric}...\n")

    # Find all result files in the directory (one per k value)
    paths = []
    for fname in os.listdir(dir):
        if fname.endswith(".npz"):
            path = os.path.join(dir, fname)
            paths.append(path)
            logger.info(f"Found result file: {path}")
    paths.sort()

    # Construct output filename, replacing metrics substring with metric name
    stem = Path(paths[0]).stem
    stem_no_k = stem.split("_k-")[0]
    stem_no_k = stem_no_k.replace(get_metric_substring(stem_no_k, metric), metric)
    fname = "xp-sweep-prdc-k_" + stem_no_k
    logger.info(f"Derived LaTeX table name: {fname}")

    # Map metric names to columns of interest
    cols_for_metric = METRIC_COLS.get(metric)
    if cols_for_metric is None:
        raise ValueError(f"Unsupported metric: {metric}")

    # Determine which column to display (precision or knn-filter-p)
    display_col = "precision" if metric == "prdc" else "knn-filter-p"

    # Get results per k, format cells, and build the final DataFrame
    records = []
    for path in paths:
        # Extract k value and use it for sorting
        k_val = get_k_substring(Path(path).stem).split("-")[1]
        k_sort = pd.to_numeric(k_val, errors="coerce")
        k_val = r"$\sqrt{n}$" if k_val == "None" else k_val
        logger.info(f"Fetching results for k = {k_val}")

        # Load data using module-level function
        df_mean, df_std = load_mean_std(path)

        # Filter to only columns of interest for this metric
        cols_to_keep = [c for c in cols_for_metric if c in df_mean.columns]
        df_mean = df_mean[cols_to_keep]
        df_std = df_std[[c for c in cols_to_keep if c in df_std.columns]]

        # Rename columns using METRIC_RENAME
        df_mean = df_mean.rename(columns=METRIC_RENAME)
        df_std = df_std.rename(columns=METRIC_RENAME)
        logger.info(f"Mean DataFrame: \n{df_mean}")
        logger.info(f"Std DataFrame: \n{df_std}")

        # Update display_col to use renamed column name
        display_col_renamed = METRIC_RENAME.get(display_col, display_col)

        # diff: overall - agg (mean only, no std), matching the xp table
        df_mean.loc["diff", display_col_renamed] = (
            df_mean.loc["overall", display_col_renamed]
            - df_mean.loc["agg", display_col_renamed]
        )
        df_std.loc["diff", display_col_renamed] = np.nan

        records.append(
            {
                "k": k_val,
                "k_sort": k_sort,
                "overall": fmt_cell("overall", display_col_renamed, df_mean, df_std),
                "avg": fmt_cell("agg", display_col_renamed, df_mean, df_std),
                "diff": fmt_cell("diff", display_col_renamed, df_mean, df_std),
            }
        )

    df_final = (
        pd.DataFrame(records)
        .sort_values("k_sort")
        .drop(columns="k_sort")
        .set_index("k")
        .rename(
            columns={
                "overall": PRECISION_EST_STR,
                "avg": AVG_COND_PRECISION_EST_STR,
                "diff": DIFF_STR,
            }
        )
    )
    df_final.index.name = "k"

    # --- Save LaTeX table ---

    format_kwargs = {"na_rep": ""}
    to_latex_kwargs = {"column_format": "c" * (len(df_final.columns) + 1)}

    if outdir is not None:
        outdir = Path(outdir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        save_latex_table(
            df_final,
            outdir=outdir,
            fname=fname,
            format_kwargs=format_kwargs,
            to_latex_kwargs=to_latex_kwargs,
        )

    return df_final, fname


####################################################################################################
# Labelwise
####################################################################################################


@main.command()
@click.option("--path-true", "-pt",     help="Path containing the results for true labels",                 type=click.Path(exists=True))  # fmt: skip
@click.option("--path-rand", "-pr",     help="Path containing the results for randomized labels ",          type=click.Path(exists=True))  # fmt: skip
@click.option("--metric", "-m",         help="Metrics to include. Default: all found",                      type=str, multiple=True, required=True)  # fmt: skip
@click.option("--outdir", "-o",         help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="out-latexify")  # fmt: skip
def labelwise_vs_rand(path_true, path_rand, metric, outdir):
    logger.info(
        "\n"
        f"Processing labelwise results in {path_true}..."
        f"Processing labelwise results in {path_rand}..."
        f"Metric: {list(metric)}"
    )

    # Quick sanity check
    metrics = list(metric)

    stem_true = Path(path_true).stem
    stem_rand = Path(path_rand).stem
    assert stem_true == stem_rand.replace("random-labs_", "")

    for m in metrics:
        assert m in stem_true, f"Metric '{m}' not found in filename '{stem_true}'"
        assert m in stem_rand, f"Metric '{m}' not found in filename '{stem_rand}'"

    for metric in metrics:
        _labelwise_vs_rand(path_true, path_rand, metric, outdir)


def _labelwise_vs_rand(path_true, path_rand, metric, outdir):

    # Build output fname
    stem = Path(path_true).stem
    metric_substring = get_metric_substring(stem, metric)
    fname = "labelwise-vs-rand_" + stem.replace(metric_substring, metric)
    logger.info(f"Derived LaTeX table name: {fname}")

    # --- Select columns to display ---

    # Maps metric names to their column names(used for filtering and ordering)
    metrics_map = {
        "prdc": [
            "prdc_nreal",
            "prdc_nfake",
            "precision",
            # "recall",
        ],
        "knn-filter": [
            "knn-filter-nreal",
            "knn-filter-nfake",
            "knn-filter-p",
            # "knn-filter-r",
        ],
    }
    target_cols = metrics_map.get(metric, [metric.upper()])

    # Load data (raw, without renaming)
    df_tl_mean, df_tl_std = load_mean_std(path_true)
    df_rl_mean, df_rl_std = load_mean_std(path_rand)

    # Filter dataframes to only include selected raw metric columns
    cols = [col for col in target_cols if col in df_rl_mean.columns]
    df_rl_mean = df_rl_mean[cols]
    df_tl_mean = df_tl_mean[cols]
    df_rl_std = df_rl_std[[c for c in cols if c in df_rl_std.columns]]
    df_tl_std = df_tl_std[[c for c in cols if c in df_tl_std.columns]]

    # --- Select rows to display ---

    # Get number of labels
    num_labels_rl = sum(1 for r in df_rl_mean.index if r.startswith("label-"))
    num_labels_tl = sum(1 for r in df_tl_mean.index if r.startswith("label-"))
    assert num_labels_rl == num_labels_tl
    num_labels = num_labels_tl

    # Get labelwise rows, keep only the 5 best and 5 worst by the selected metric
    ordering_cols = [col for col in cols if "nreal" not in col and "nfake" not in col]
    ordering_score = [
        df_tl_mean.loc[f"label-{i}", ordering_cols[0]] for i in range(num_labels)
    ]
    selected_labels = select_best_worst_indexes(ordering_score, n_each=5)

    # Get rows to display
    label_rows = [f"label-{i}" for i in selected_labels]
    all_rows = ["overall"] + label_rows + ["agg"]

    # --- Groups ---

    # Count block
    count_cols = [c for c in cols if "nreal" in c or "nfake" in c]
    score_cols = [c for c in cols if c not in count_cols]

    counts_tl = fmt_block(df_tl_mean, df_tl_std, count_cols, all_rows)
    counts_rl = fmt_block(df_rl_mean, df_rl_std, count_cols, all_rows)
    assert counts_tl.equals(counts_rl), f"Counts:\n{counts_tl}\n{counts_rl}"

    # Build count block with proper MultiIndex: (Counts, column_name)
    block_counts = counts_tl
    block_counts.columns = pd.MultiIndex.from_product([["Counts"], count_cols])

    # Results block
    block_rl = fmt_block(df_rl_mean, df_rl_std, score_cols, all_rows)
    block_tl = fmt_block(df_tl_mean, df_tl_std, score_cols, all_rows)

    # Delta block
    # is meaningful only on the aggregate row: E_Y[rand] - E_Y[cond] (mean only)
    diff = df_rl_mean.loc["agg", score_cols] - df_tl_mean.loc["agg", score_cols]
    block_diff = pd.DataFrame("", index=all_rows, columns=score_cols)
    block_diff.loc["agg"] = [f"{float(diff[c]):.3f}" for c in score_cols]

    # Build metric blocks with proper MultiIndex: (group, column_name)
    block_rl.columns = pd.MultiIndex.from_product([["Rand. label $Z$"], score_cols])
    block_tl.columns = pd.MultiIndex.from_product([["Cond. label $Y$"], score_cols])
    block_diff.columns = pd.MultiIndex.from_product([["$\Delta$"], score_cols])

    # Combine all blocks
    df_main = pd.concat([block_counts, block_rl, block_tl, block_diff], axis=1)

    # Rename cols and rows
    col_rename = {col: METRIC_RENAME.get(col, None) for col in cols}
    row_rename = {
        "overall": PRECISION_EST_STR,
        "agg": AVG_COND_PRECISION_EST_STR.replace("Y", r"\cdot"),
        **{f"label-{i}": RAND_COND_PRECISION_EST_STR(i) for i in selected_labels},
    }
    df_final = df_main.rename(index=row_rename).rename(columns=col_rename)

    # --- Save LaTeX table ---

    format_kwargs = {"na_rep": ""}
    to_latex_kwargs = {"column_format": "c" * (len(df_final.columns) + 1)}

    if outdir is not None:
        outdir = Path(outdir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        save_latex_table(
            df_final,
            outdir=outdir,
            fname=fname,
            format_kwargs=format_kwargs,
            to_latex_kwargs=to_latex_kwargs,
        )

    return df_final, fname


if __name__ == "__main__":
    main()
