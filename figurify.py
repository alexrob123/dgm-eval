import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from dgm_eval.metrics.pr_curve import PR_CURVE_CLFS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)


plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.left"] = True
plt.rcParams["axes.spines.bottom"] = True
plt.rcParams["axes.grid"] = False
plt.rcParams["grid.alpha"] = 0.2
plt.rcParams["font.size"] = 16
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["xaxis.labellocation"] = "center"
plt.rcParams["yaxis.labellocation"] = "center"
plt.rcParams["legend.fontsize"] = "x-small"
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern Roman"],
    }
)

PR_CURVE_DIR = "./out"
PR_CURVE_METRICS = ["pr_curve_" + clf for clf in PR_CURVE_CLFS]


# Mathematical notation for labels
PRECISION_EST_STR = r"$\hat \alpha (P_X, Q_X)$"
AVG_COND_PRECISION_EST_STR = r"$E_Y \left[ \hat \alpha(P_{X|Y}, Q_{X|Y}) \right]$"
COND_PRECISION_EST_STR = r"$\hat \alpha(P_{X|Y=i}, Q_{X|Y=i})$"


def cond_precision_str(i):
    return rf"$\hat \alpha (P_{{X|Y={{{i}}}}}, Q_{{X|Y={{{i}}}}})$"


####################################################################################################
# Utility Functions
####################################################################################################


def _extract_single_curve(run_results, metric_key, result_key, std_key=None):
    """Extract a single PR curve with optional std from run results.

    Parameters
    ----------
    run_results : dict
        Result dictionary from npz file
    metric_key : str
        The metric key to extract (e.g., 'pr-curve-knn')
    result_key : str
        The key in run_results to extract from (e.g., 'overall', 'agg', 'label-0')
    std_key : str, optional
        The key for std results (e.g., 'overall_std', 'label-0_std')

    Returns
    -------
    dict or None
        Dictionary with 'recalls', 'precisions', 'std' keys (sorted by recall),
        or None if metric_key not found in result_key
    """
    result = run_results.get(result_key, {})
    if metric_key not in result:
        return None

    metric = result[metric_key]
    precisions = metric["P"]
    recalls = metric["R"]
    sort_idx = np.argsort(recalls)

    curve_data = {
        "recalls": recalls[sort_idx],
        "precisions": precisions[sort_idx],
        "std": None,
    }

    # Extract std if available
    if std_key and std_key in run_results and metric_key in run_results[std_key]:
        prec_std = run_results[std_key][metric_key]["P"]
        curve_data["std"] = prec_std[sort_idx]

    return curve_data


def _plot_curve_with_std(ax, recalls, precisions, std, color, alpha, linewidth, label):
    """Plot a curve with optional shaded std area.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    recalls : np.ndarray
        Recall values (x-axis)
    precisions : np.ndarray
        Precision values (y-axis)
    std : np.ndarray or None
        Standard deviation values
    color : str
        Line color
    alpha : float
        Line alpha
    linewidth : float
        Line width
    label : str
        Legend label
    """
    if std is not None:
        ax.fill_between(
            recalls,
            np.maximum(0, precisions - std),
            np.minimum(1, precisions + std),
            alpha=alpha * 0.5,
            color=color,
            zorder=1,
        )

    ax.plot(
        recalls,
        precisions,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
        zorder=2,
    )


####################################################################################################
# PR Curve Plotting
####################################################################################################


def _extract_pr_curves(run_results, metric_key):
    """Extract PR curves data from results.

    Returns
    -------
    dict
        Dictionary with keys: 'overall', 'labels', 'agg', each containing
        {'recalls': sorted recalls, 'precisions': sorted precisions, 'std': std (if available)}
    """
    curves = {}

    # Extract overall pr-curve
    overall = _extract_single_curve(run_results, metric_key, "overall", "overall_std")
    if overall:
        curves["overall"] = overall

    # Extract per-label pr-curves
    label_keys = sorted(
        [k for k in run_results if k.startswith("label-") and not k.endswith("_std")],
        key=lambda x: int(x.split("-")[1]),
    )

    curves["labels"] = {}
    for label_key in label_keys:
        label = _extract_single_curve(
            run_results, metric_key, label_key, f"{label_key}_std"
        )
        if label:
            curves["labels"][label_key] = label

    # Extract aggregated pr-curve
    agg = _extract_single_curve(run_results, metric_key, "agg", "agg_std")
    if agg:
        curves["agg"] = agg

    return curves


def _create_pr_figure(curves, curves_rand=None, label_flag=True):
    """Create PR curve figure from extracted data.

    Parameters
    ----------
    curves : dict
        Dictionary with PR curves data from _extract_pr_curves
    curves_rand : dict, optional
        Dictionary with PR curves data from random labels. If provided,
        aggregated random labels curve will be added to the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # --- Curves for random label reference ---
    # if provided so they appear below actual curves
    # label-i: blue
    # agg: navy
    # overall: green

    # Plot per-label curves from random labels if provided
    if curves_rand is not None and label_flag and "labels" in curves_rand:
        label_keys = sorted(curves_rand["labels"].keys())

        # If more than 10 labels, randomly select 10
        if len(label_keys) > 10:
            rng = np.random.default_rng(seed=0)  # For reproducibility
            selected_keys = sorted(rng.choice(label_keys, 10, replace=False))
        else:
            selected_keys = label_keys

        label_desc = "Individual per-label (random labels)"

        for label_key in selected_keys:
            data = curves_rand["labels"][label_key]
            _plot_curve_with_std(
                ax,
                data["recalls"],
                data["precisions"],
                data["std"],
                color="blue",
                alpha=0.3,
                linewidth=0.8,
                label=None,
            )

        # Add legend entry for per-label curves
        ax.plot(
            [],
            [],
            color="blue",
            alpha=0.3,
            linewidth=0.8,
            label=label_desc,
        )

    # Plot aggregated curve from random labels if provided
    if curves_rand is not None and "agg" in curves_rand:
        data = curves_rand["agg"]
        _plot_curve_with_std(
            ax,
            data["recalls"],
            data["precisions"],
            data["std"],
            color="blue",
            alpha=1.0,
            linewidth=2.0,
            label="Aggregated per-label (random labels)",
        )

    # Plot overall curve from random labels if provided
    if curves_rand is not None and "overall" in curves_rand:
        data = curves_rand["overall"]
        _plot_curve_with_std(
            ax,
            data["recalls"],
            data["precisions"],
            data["std"],
            color="navy",
            alpha=1.0,
            linewidth=2.5,
            label="Overall (random labels)",
        )

    # --- Actual curves ---
    # label-i: red
    # agg: firebrick
    # overall: green

    # Plot per-label curves
    if label_flag and "labels" in curves:
        label_keys = sorted(curves["labels"].keys())

        # If more than 10 labels, randomly select 10
        if len(label_keys) > 10:
            rng = np.random.default_rng(seed=0)  # For reproducibility
            selected_keys = sorted(rng.choice(label_keys, 10, replace=False))
        else:
            selected_keys = label_keys

        label_desc = "Individual per-label"

        for label_key in selected_keys:
            data = curves["labels"][label_key]
            _plot_curve_with_std(
                ax,
                data["recalls"],
                data["precisions"],
                # data["std"],
                None,
                color="red",
                alpha=0.3,
                linewidth=0.8,
                label=None,
            )

        # Add legend entry for per-label curves
        ax.plot(
            [],
            [],
            color="yellow",
            alpha=0.3,
            linewidth=0.8,
            label=label_desc,
        )

    # Plot aggregated curve
    if "agg" in curves:
        data = curves["agg"]
        _plot_curve_with_std(
            ax,
            data["recalls"],
            data["precisions"],
            data["std"],
            color="orange",
            alpha=1.0,
            linewidth=2.0,
            label="Aggregated per-label",
        )

    # Plot overall curve last so it appears on top
    if "overall" in curves:
        data = curves["overall"]
        _plot_curve_with_std(
            ax,
            data["recalls"],
            data["precisions"],
            data["std"],
            color="red",
            alpha=1.0,
            linewidth=2.5,
            label="Overall",
        )

    # --- Formatting ---

    # Reorder legend (hardcoded order)
    handles, labels = ax.get_legend_handles_labels()

    # Define desired order
    desired_order = [
        "Overall",
        "Individual per-label",
        "Aggregated per-label",
        "Overall (random labels)",
        "Individual per-label (random labels)",
        "Aggregated per-label (random labels)",
    ]

    # Reorder handles and labels based on desired order
    order_indices = []
    for desired_label in desired_order:
        for i, label in enumerate(labels):
            if label == desired_label and i not in order_indices:
                order_indices.append(i)
                break

    handles = [handles[i] for i in order_indices]
    labels = [labels[i] for i in order_indices]

    # Rename labels with mathematical notation
    new_labels = []
    for label in labels:
        if label == "Overall":
            new_labels.append(PRECISION_EST_STR)
        elif label == "Overall (random labels)":
            new_labels.append(PRECISION_EST_STR.replace("Y", "Z"))
        elif label == "Individual per-label":
            new_labels.append(COND_PRECISION_EST_STR)
        elif label == "Individual per-label (random labels)":
            new_labels.append(COND_PRECISION_EST_STR.replace("Y", "Z"))
        elif label == "Aggregated per-label":
            new_labels.append(AVG_COND_PRECISION_EST_STR)
        elif label == "Aggregated per-label (random labels)":
            new_labels.append(AVG_COND_PRECISION_EST_STR.replace("Y", "Z"))
        else:
            new_labels.append(label)

    ax.legend(handles, new_labels, fontsize=10, loc="best")

    # Formatting
    ax.set_xlabel(r"Recall ($\beta$)", fontsize=12)
    ax.set_ylabel(r"Precision ($\alpha$)", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    return fig, ax


def plot_pr_curve(
    path,
    metric_key="pr_curve_knn",
    outdir=None,
    path_rand=None,
    label_flag=True,
):
    """Plot Precision-Recall curve with overall (blue) and per-label (red) curves.

    Parameters
    ----------
    path : str | Path
        Path to the result .npz file containing pr-curve results
    metric_key : str
        The pr_curve metric key to plot (e.g., 'pr_curve_knn', 'pr_curve_ipr')
    outdir : str | Path, optional
        Directory to save the figure. If None, displays the plot.
    path_rand : str | Path, optional
        Path to the result .npz file from random labels experiment. If provided,
        curves will be added to the plot with labelwise line in blue.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Load and extract data
    data = np.load(path, allow_pickle=True)
    run_results = data["scores"].item()["run00"]
    curves = _extract_pr_curves(run_results, metric_key)

    # Validate random labels data if provided and load
    curves_rand = None
    if path_rand is not None:
        stem = Path(path).stem
        stem_rand = Path(path_rand).stem
        stem_rand_drop = stem_rand.replace("-random_labs", "")
        if stem != stem_rand_drop:
            logger.warning(
                f"Filenames don't match after removing '-random_labs': "
                f"'{stem}' vs '{stem_rand_drop}'. "
                f"Experiment setup may be different."
            )

        data_rand = np.load(path_rand, allow_pickle=True)
        run_results_rand = data_rand["scores"].item()["run00"]
        curves_rand = _extract_pr_curves(run_results_rand, metric_key)

    # Create figure
    fig, ax = _create_pr_figure(curves, curves_rand, label_flag)

    # Set title
    clf = metric_key.split("_")[-1].upper()
    ax.set_title(f"PR Curve ({clf})", fontsize=13, fontweight="bold")

    # Save or display
    if outdir is not None:
        outdir = Path(outdir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        fname = Path(path).stem

        # Save both PDF (for LaTeX) and PNG (for viewing)
        for fmt in [
            # "pdf",
            "png",
        ]:
            out_path = outdir / f"pr-curve_{fname}.{fmt}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", format=fmt)
            logger.info(f"Saved PR curve figure to {out_path}")
    else:
        plt.show()

    return fig


####################################################################################################
# CLI
####################################################################################################


@click.group()
def main():
    """Process results and generate figures."""


@main.command()
@click.option("--path", "-p",          help="Path to result .npz file",                          type=click.Path(exists=True), default=None)  # fmt: skip
@click.option("--path-rand", "-r",     help="Path to random labels .npz file",                   type=click.Path(exists=True), default=None)  # fmt: skip
@click.option("--metric", "-m",        help="Metric key (e.g., 'pr_curve_knn', 'pr_curve_ipr')", type=str, default=None)  # fmt: skip
@click.option("--outdir", "-o",        help="Directory to save the figure",                      type=click.Path(), default="out-figurify")  # fmt: skip
@click.option("--plt-label", "-l",     help="Flag to display label-wise curves",                 is_flag=True, default=True)  # fmt: skip
def pr_curve(path, metric, outdir, path_rand, plt_label):

    if path is None:
        for path_, path_rand_, metric_ in pr_curve_default(PR_CURVE_DIR, metric):
            logger.info(f"Plotting PR curve from {path_} (metric: {metric_})...")
            pr_curve.callback(str(path_), metric_, outdir, str(path_rand_), plt_label)
        return

    logger.info(f"Plotting PR curve from {path} (metric: {metric})...")

    if path_rand:
        logger.info(f"Adding random labels curves from {path_rand}...")

    plot_pr_curve(
        path,
        metric_key=metric,
        outdir=outdir,
        path_rand=path_rand,
        label_flag=plt_label,
    )


def pr_curve_default(base_dir, metric=None):
    """Find all (path_true, path_rand, metric) triplets for pr-curve files in base_dir."""
    base_dir = Path(base_dir)

    all_npz = [f for f in base_dir.glob("*.npz") if "pr_curve" in f.name]

    rand_files = {
        f.name.replace("-random_labs", ""): f
        for f in all_npz
        if "random_labs" in f.name
    }
    true_files = {f.name: f for f in all_npz if "random_labs" not in f.name}

    pairs = [(true_files[k], rand_files[k]) for k in true_files if k in rand_files]
    if not pairs:
        logger.warning("No matching pr-curve pairs found.")
        return []

    metrics_to_try = [metric] if metric else PR_CURVE_METRICS

    triplets = []
    for path_true, path_rand in pairs:
        for m in metrics_to_try:
            if m in path_true.name:
                triplets.append((path_true, path_rand, m))

    return triplets


if __name__ == "__main__":
    main()
