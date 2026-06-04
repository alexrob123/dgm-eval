import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s", force=True
)

METRIC_RENAME = {
    "precision": "P",
    "recall": "R",
    "density": "D",
    "coverage": "C",
    "prdc_nreal": "N real",
    "prdc_nfake": "N fake",
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


def fmt_cell(row, col, df_mean, df_std, count_col):
    mean = df_mean.loc[row, col]
    if col == count_col:
        return f"{int(mean)}" if pd.notna(mean) else ""
    std = df_std.loc[row, col] if col in df_std.columns else np.nan
    if pd.isna(std):
        return f"{mean:.3f}\phantom{{$~\pm~0.000$}}"
    return f"{mean:.3f} $\pm$ {std:.3f}"


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


def save_latex_table(
    df: pd.DataFrame,
    outdir: str | Path,
    fname: str,
    format_kwargs=None,
    to_latex_kwargs=None,
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
            col: [
                fmt_cell(row, col, df_mean, df_std, count_col) for row in df_mean.index
            ]
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
# XP KNN BALLS FILTERING
####################################################################################################


@main.command()
@click.option("--dir", "-d",        help="Directory containing the sweep results",                      type=click.Path(exists=True))  # fmt: skip
@click.option("--outdir", "-o",     help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="out-latexify")  # fmt: skip
def xp_filter_knn_balls(dir, outdir):
    logger.info(f"Processing kNN balls filtering results in {dir}...")

    paths = []
    for fname in os.listdir(dir):
        if fname.endswith(".npz"):
            path = os.path.join(dir, fname)
            paths.append(path)
            logger.info(f"Found result file: {path}")
    paths.sort()

    fname = "xp-knn-balls-filtering-k_" + Path(paths[0]).stem.split("_k-")[0]
    logger.info(f"Derived LaTeX table name: {fname}")

    count_col = "prdc_nreal"  # prdc_nreal == prdc_nfake always
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
                "overall": fmt_cell("overall", metric, df_mean, df_std, count_col),
                "avg": fmt_cell("avg", metric, df_mean, df_std, count_col),
                "diff": fmt_cell("diff", metric, df_mean, df_std, count_col),
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
@click.option("--outdir", "-o",     help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="out-latexify")  # fmt: skip
def xp_sweep_prdc_k(dir, outdir):
    logger.info(f"Processing sweep results in {dir}...")

    paths = []
    for fname in os.listdir(dir):
        if fname.endswith(".npz"):
            path = os.path.join(dir, fname)
            paths.append(path)
            logger.info(f"Found result file: {path}")
    paths.sort()

    fname = "xp-sweep-prdc-k_" + Path(paths[0]).stem.split("_k-")[0]
    logger.info(f"Derived LaTeX table name: {fname}")

    count_col = "prdc_nreal"  # prdc_nreal == prdc_nfake always
    metric = "P"  # report precision only

    records = []  # one row per k: overall / avg / diff precision
    for path in paths:
        k_val = Path(path).stem.split("-")[-1]  # "1", "2", "None"
        k_sort = pd.to_numeric(k_val, errors="coerce")
        k_val = r"$\sqrt{n}$" if k_val == "None" else k_val
        logger.info(f"Fetching results for k = {k_val}")

        data = np.load(path, allow_pickle=True)
        df = pd.DataFrame(data["scores"].item()["run00"]).T

        # Split mean / std rows (std = variance across runs, i.e. prdc subsampling)
        mean_rows = [r for r in df.index if not r.endswith("_std")]
        std_rows = [r for r in df.index if r.endswith("_std")]
        df_mean = df.loc[mean_rows].rename(columns=METRIC_RENAME).copy()
        df_std = (
            df.loc[std_rows]
            .rename(index=lambda x: x.removesuffix("_std"), columns=METRIC_RENAME)
            .copy()
        )

        # diff: overall - agg (mean only, no std), matching the xp table
        df_mean.loc["diff", metric] = (
            df_mean.loc["overall", metric] - df_mean.loc["agg", metric]
        )
        df_std.loc["diff", metric] = np.nan

        records.append(
            {
                "k": k_val,
                "k_sort": k_sort,
                "overall": fmt_cell("overall", metric, df_mean, df_std, count_col),
                "avg": fmt_cell("agg", metric, df_mean, df_std, count_col),
                "diff": fmt_cell("diff", metric, df_mean, df_std, count_col),
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
# XP RANDOMIZE LABELS
####################################################################################################


@main.command()
@click.option("--path-rand", "-pr",     help="Path containing the results for randomized labels ",          type=click.Path(exists=True))  # fmt: skip
@click.option("--path-true", "-pt",     help="Path containing the results for true labels",                 type=click.Path(exists=True))  # fmt: skip
@click.option("--outdir", "-o",         help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="out-latexify")  # fmt: skip
def tab_labelwise(path_rand, path_true, outdir):
    logger.info(f"Processing labelwise results in {path_rand}...")

    if "filter-knn-balls" in path_rand:
        logger.info(
            "Detected 'filter-knn-balls' in path; using it to derive the table name."
        )

    fname = "tab-labelwise" + Path(path_rand).stem
    logger.info(f"Derived LaTeX table name: {fname}")

    count_col = "prdc_nreal"  # prdc_nreal == prdc_nfake always

    def load_mean_std(path):
        """Load a result file and split it into mean / std DataFrames.

        Scores carry per-label rows plus their "*_std" counterparts (variance
        across runs). For randomized labels that variance is over the random
        label draws; for true labels it is over the prdc subsampling.
        """
        data = np.load(path, allow_pickle=True)
        df = pd.DataFrame(data["scores"].item()["run00"]).T
        mean_rows = [r for r in df.index if not r.endswith("_std")]
        std_rows = [r for r in df.index if r.endswith("_std")]
        df_mean = df.loc[mean_rows].rename(columns=METRIC_RENAME).copy()
        df_std = (
            df.loc[std_rows]
            .rename(index=lambda x: x.removesuffix("_std"), columns=METRIC_RENAME)
            .copy()
        )
        return df_mean, df_std

    # Randomized labels (rl) vs true / conditioning labels (tl)
    df_rl_mean, df_rl_std = load_mean_std(path_rand)
    df_tl_mean, df_tl_std = load_mean_std(path_true)

    num_labels_rl = sum(1 for r in df_rl_mean.index if r.startswith("label-"))
    num_labels_tl = sum(1 for r in df_tl_mean.index if r.startswith("label-"))
    assert num_labels_rl == num_labels_tl
    num_labels = num_labels_tl

    # Display the 5 best and 5 worst labels by precision under the conditioning
    # label Y (true labels), ordered by index -- not by the randomized label Z.
    label_scores = [df_tl_mean.loc[f"label-{i}", "P"] for i in range(num_labels)]
    selected_labels = select_best_worst_indexes(label_scores, n_each=5)
    label_rows = [f"label-{i}" for i in selected_labels]
    body_rows = ["overall"] + label_rows + ["agg"]

    metric_order = ["P"]
    group_order = [r"Rand. label $Z$", r"Cond. label $Y$", r"$\Delta$"]

    def fmt_block(df_mean, df_std):
        """Format one source into "mean $\\pm$ std" cells for the body rows."""
        return pd.DataFrame(
            {
                col: [fmt_cell(r, col, df_mean, df_std, count_col) for r in body_rows]
                for col in metric_order
            },
            index=body_rows,
        )

    block_rl = fmt_block(df_rl_mean, df_rl_std)
    block_tl = fmt_block(df_tl_mean, df_tl_std)

    # Delta is meaningful only on the aggregate row: E_Y[rand] - E_Y[cond] (mean only)
    diff = df_rl_mean.loc["agg", metric_order] - df_tl_mean.loc["agg", metric_order]
    block_diff = pd.DataFrame("", index=body_rows, columns=metric_order)
    block_diff.loc["agg"] = [f"{float(diff[c]):.3f}" for c in metric_order]

    df_main = pd.concat(
        {
            r"Rand. label $Z$": block_rl,
            r"Cond. label $Y$": block_tl,
            r"$\Delta$": block_diff,
        },
        axis=1,
    ).reindex(columns=pd.MultiIndex.from_product([group_order, metric_order]))

    row_rename = {
        "overall": PRECISION_EST_STR,
        "agg": AVG_COND_PRECISION_EST_STR.replace("Y", r"\cdot"),
        **{f"label-{i}": RAND_COND_PRECISION_EST_STR(i) for i in selected_labels},
    }
    index_order = [row_rename[r] for r in body_rows]
    df_final = df_main.rename(index=row_rename).reindex(index=index_order)

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
