import logging
import os
from copy import deepcopy
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
}


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
# XP KNN BALLS FILTERING
####################################################################################################


@main.command()
@click.option("--dir", "-d",        help="Directory containing the sweep results",                      type=click.Path(exists=True))  # fmt: skip
@click.option("--outdir", "-o",     help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="out-latexify")  # fmt: skip
def xp_knn_balls_filtering(dir, outdir):
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

    records = []
    for path in paths:
        k_val = Path(path).stem.split("-")[-1]  # "1", "2", "None"
        k_val = r"$\sqrt{n}$" if k_val == "None" else k_val
        logger.info(f"Fetching results for k = {k_val}")

        data = np.load(path, allow_pickle=True)
        metrics = data["scores"].item()["run00"]

        # label metrics are under overall, so we pop them out and put them in the main dict
        metrics_cpy = deepcopy(metrics)
        metrics_upd = {}
        for k in metrics["overall"].keys():
            if "label" in k:
                metrics_upd[k] = metrics_cpy["overall"].pop(k)
        metrics_upd["overall"] = metrics_cpy["overall"]
        metrics = metrics_upd

        num_labels = len(metrics) - 1  # excluding "overall"
        logger.info(f"Number of labels (excluding overall): {num_labels}")
        for label, vals in metrics.items():
            records.append({"k": k_val, "label": label, **vals})

    df_results = pd.DataFrame(records)

    # Optional: sort and make k numeric where possible
    df_results["k_sort"] = pd.to_numeric(df_results["k"], errors="coerce")
    df_results = (
        df_results.sort_values(["k_sort", "label"])
        .drop(columns="k_sort")
        .reset_index(drop=True)
    )

    df_overall = (
        df_results[df_results["label"] == "overall"]
        .drop(columns="label")
        .set_index("k")
    )
    df_labels = df_results.drop(df_results[df_results["label"] == "overall"].index)
    averages = {}
    for k in df_labels["k"].unique():
        avg = df_labels[df_labels["k"] == k].drop(columns="k").set_index("label").mean()
        averages[k] = avg
    df_averages = pd.DataFrame(averages).T
    df_averages.index.name = "k"

    df_diff = df_overall - df_averages

    df_combined = pd.concat(
        {"overall": df_overall, "avg": df_averages, "diff": df_diff},
        axis=1,
    ).swaplevel(0, 1, axis=1)  # now (metric, source)

    source_rename = {
        "overall": r"$\hat \alpha (P_X, Q_X)$",
        "avg": r"$E_{P_Y} \left[ \hat \alpha(P_{X|Y}, Q_{X|Y}) \right]$",
        "diff": r"$\Delta$",
    }
    metric_order = [
        "P",
        "R",
        # "D",
        # "C",
    ]
    source_order = [source_rename[k] for k in ("overall", "avg", "diff")]

    df_combined = (
        df_combined.rename(columns=METRIC_RENAME, level=0)
        .rename(columns=source_rename, level=1)
        .reindex(columns=pd.MultiIndex.from_product([metric_order, source_order]))
    )

    df_final = df_combined

    format_kwargs = {}
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

    records = []
    for path in paths:
        k_val = Path(path).stem.split("-")[-1]  # "1", "2", "None"
        k_val = r"$\sqrt{n}$" if k_val == "None" else k_val
        logger.info(f"Fetching results for k = {k_val}")

        data = np.load(path, allow_pickle=True)
        metrics = data["scores"].item()["run00"]
        num_labels = len(metrics) - 1  # excluding "overall"
        logger.info(f"Number of labels (excluding overall): {num_labels}")
        for label, vals in metrics.items():
            records.append({"k": k_val, "label": label, **vals})

    df_results = pd.DataFrame(records)

    # Optional: sort and make k numeric where possible
    df_results["k_sort"] = pd.to_numeric(df_results["k"], errors="coerce")
    df_results = (
        df_results.sort_values(["k_sort", "label"])
        .drop(columns="k_sort")
        .reset_index(drop=True)
    )

    df_overall = (
        df_results[df_results["label"] == "overall"]
        .drop(columns="label")
        .set_index("k")
    )
    df_labels = df_results.drop(df_results[df_results["label"] == "overall"].index)
    averages = {}
    for k in df_labels["k"].unique():
        avg = df_labels[df_labels["k"] == k].drop(columns="k").set_index("label").mean()
        averages[k] = avg
    df_averages = pd.DataFrame(averages).T
    df_averages.index.name = "k"

    df_diff = df_overall - df_averages

    df_combined = pd.concat(
        {"overall": df_overall, "avg": df_averages, "diff": df_diff},
        axis=1,
    ).swaplevel(0, 1, axis=1)  # now (metric, source)

    source_rename = {
        "overall": r"$\hat \alpha (P_X, Q_X)$",
        "avg": r"$E_{P_Y} \left[ \hat \alpha(P_{X|Y}, Q_{X|Y}) \right]$",
        "diff": r"$\Delta$",
    }
    metric_order = [
        "P",
        "R",
        # "D",
        # "C",
    ]
    source_order = [source_rename[k] for k in ("overall", "avg", "diff")]

    df_combined = (
        df_combined.rename(columns=METRIC_RENAME, level=0)
        .rename(columns=source_rename, level=1)
        .reindex(columns=pd.MultiIndex.from_product([metric_order, source_order]))
    )

    df_final = df_combined

    format_kwargs = {}
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
def xp_randomize_labels(path_rand, path_true, outdir):
    logger.info(f"Processing randomize labels results in {path_rand}...")

    fname = "xp-randomize-labels_" + Path(path_rand).stem
    logger.info(f"Derived LaTeX table name: {fname}")

    # Results with true labels (tl)
    data_tl = np.load(path_true, allow_pickle=True)
    metrics_tl = data_tl["scores"].item()["run00"]
    num_labels_tl = len(metrics_tl) - 1  # excluding "overall"

    # Results with randomized labels (rl)
    data_rl = np.load(path_rand, allow_pickle=True)
    metrics_rl = data_rl["scores"].item()["run00"]
    num_labels_rl = len(metrics_rl) - 1  # excluding "overall"

    assert num_labels_tl == num_labels_rl
    num_labels = num_labels_tl

    # Num labels to display in the table
    max_display_labels = 10
    if num_labels > max_display_labels:
        rng = np.random.default_rng(0)
        selected_labels = sorted(
            rng.choice(num_labels, size=max_display_labels, replace=False).tolist()
        )
        logger.info(
            f"num_labels={num_labels} > {max_display_labels}; "
            f"sampling labels {selected_labels} for display"
        )
    else:
        selected_labels = list(range(num_labels))

    df_tl = pd.DataFrame(metrics_tl).T.rename(columns=METRIC_RENAME)
    df_rl = pd.DataFrame(metrics_rl).T.rename(columns=METRIC_RENAME)

    metric_order = ["P", "D"]
    group_order = ["Randomized label", "Conditioning label", r"$\Delta$"]

    # Aggregate (mean) over all per-label rows, excluding "overall"
    avg_tl = df_tl.drop(index="overall").mean()
    avg_rl = df_rl.drop(index="overall").mean()
    diff = avg_rl - avg_tl

    # Rows to display: "overall" + a sample of per-label rows
    label_rows = [f"label-{i}" for i in selected_labels]
    row_keys = ["overall"] + label_rows

    df_main = pd.concat(
        {
            "Randomized label": df_rl.loc[row_keys],
            "Conditioning label": df_tl.loc[row_keys],
        },
        axis=1,
    ).reindex(columns=pd.MultiIndex.from_product([group_order, metric_order]))

    # Aggregate (mean) row at the bottom; the delta group is filled only here
    df_main.loc["avg", "Randomized label"] = avg_rl[metric_order].to_numpy()
    df_main.loc["avg", "Conditioning label"] = avg_tl[metric_order].to_numpy()
    df_main.loc["avg", r"$\Delta$"] = diff[metric_order].to_numpy()

    desc_rename = {
        "overall": r"$\hat \alpha (P_X, Q_X)$",
        "avg": r"$E_{P_Y} \left[ \hat \alpha(P_{X|Y}, Q_{X|Y}) \right]$",
        **{
            f"label-{i}": rf"$\hat \alpha (P_{{X|Y={{{i}}}}}, Q_{{X|Y={{{i}}}}})$"
            for i in selected_labels
        },
    }
    index_order = (
        [desc_rename["overall"]]
        + [desc_rename[k] for k in label_rows]
        + [desc_rename["avg"]]
    )

    df_final = df_main.rename(index=desc_rename).reindex(index=index_order)

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
