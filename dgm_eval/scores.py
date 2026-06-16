import logging
import os
import pathlib
import sys
from collections import defaultdict
from pprint import pformat

import numpy as np
import pandas as pd

from .metrics import (
    compute_authpct,
    compute_CTscore,
    compute_CTscore_mem,
    compute_CTscore_mode,
    compute_efficient_FD_with_reps,
    compute_FD_infinity,
    compute_FD_with_reps,
    compute_fls,
    compute_fls_overfit,
    compute_mmd,
    compute_per_class_vendi_scores,
    compute_pr_curve,
    compute_prdc,
    sw_approx,
)
from .utils import extend_path, make_str

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)

LAMBDAS = np.tan(np.linspace(0, np.pi / 2, 100 + 1)[1:])


####################################################################################################
# Compute
####################################################################################################


def compute_scores(
    args,
    rr,
    rg,
    test_reps,
    labels=None,
    label_key="overall",
    seed=0,
):
    """
    reps: list of two arrays corresponding to real reps and gen reps, each of shape (Nimage, ndim)
    labels: array of shape (Nimage,) with class labels for generated samples, if available
    """

    scores = {}
    vendi_scores = {}

    if "authpct" in args.metrics and label_key == "overall":
        print("Computing authpct \n", file=sys.stderr)
        scores["authpct"] = compute_authpct(rr, rg)

    if "ct" in args.metrics and label_key == "overall":
        print("Computing ct score \n", file=sys.stderr)
        scores["ct"] = compute_CTscore(rr, test_reps, rg)

    if "ct_test" in args.metrics and label_key == "overall":
        print(
            "Computing ct score, modified to identify mode collapse only \n",
            file=sys.stderr,
        )
        scores["ct_test"] = compute_CTscore_mode(rr, test_reps, rg)

    if "ct_modified" in args.metrics and label_key == "overall":
        print(
            "Computing ct score, modified to identify memorization only \n",
            file=sys.stderr,
        )
        scores["ct_modified"] = compute_CTscore_mem(rr, test_reps, rg)

    if "denscov" in args.metrics:
        raise NotImplementedError("denscov metric is currently not implemented")

    if "fd" in args.metrics and label_key == "overall":
        print("Computing FD \n", file=sys.stderr)
        scores["fd"] = compute_FD_with_reps(rr, rg)

    if "fd_eff" in args.metrics and label_key == "overall":
        print("Computing Efficient FD \n", file=sys.stderr)
        scores["fd_eff"] = compute_efficient_FD_with_reps(rr, rg)

    if "fd_infinity" in args.metrics and label_key == "overall":
        print("Computing fd_infinity \n", file=sys.stderr)
        scores["fd_infinity_value"] = compute_FD_infinity(rr, rg)

    if set(args.metrics) & {"fls", "fls_overfit"} and label_key == "overall":
        rng = np.random.default_rng(seed)

        train_reps, gen_reps = rr, rg
        reduced_n = min(
            args.reduced_n,
            train_reps.shape[0] // 2,
            test_reps.shape[0],
            gen_reps.shape[0],
        )

        test_reps = test_reps[rng.choice(test_reps.shape[0], reduced_n, replace=False)]
        gen_reps = gen_reps[rng.choice(gen_reps.shape[0], reduced_n, replace=False)]

        print("Computing fls \n", file=sys.stderr)
        # fls must be after ot, as it changes train_reps
        train_reps = train_reps[
            rng.choice(train_reps.shape[0], 2 * reduced_n, replace=False)
        ]
        train_reps, baseline_reps = train_reps[:reduced_n], train_reps[reduced_n:]

        if "fls" in args.metrics:
            scores["fls"] = compute_fls(
                train_reps,
                baseline_reps,
                test_reps,
                gen_reps,
            )
        if "fls_overfit" in args.metrics:
            scores["fls_overfit"] = compute_fls_overfit(
                train_reps,
                baseline_reps,
                test_reps,
                gen_reps,
            )

    if "kd" in args.metrics and label_key == "overall":
        print("Computing KD \n", file=sys.stderr)
        mmd_values = compute_mmd(rr, rg)
        scores["kd_value"] = mmd_values.mean()
        scores["kd_variance"] = mmd_values.std()

    if "prdc" in args.metrics:  # compute for overall and per label
        reduced_n = min(args.reduced_n, rr.shape[0], rg.shape[0])

        logger.info(
            f"Computing PRDC with:\n\t samples = {reduced_n}\n\t k = {args.nearest_k}"
        )

        rng = np.random.default_rng(seed)
        inds0 = rng.choice(rr.shape[0], reduced_n, replace=False)
        inds1 = rng.choice(rg.shape[0], reduced_n, replace=False)

        # if "realism" not in args.metrics: # rebuild in another if, keep all generated do not downsample
        # Realism is returned for each sample, so do not shuffle if this metric is desired.
        # Else filenames and realism scores will not align
        # inds1 = rng.choice(
        #     inds1,
        #     min(inds1.shape[0], reduced_n),
        #     replace=False,
        # )

        prdc_dict = compute_prdc(rr[inds0], rg[inds1], nearest_k=args.nearest_k)
        scores.update(prdc_dict)

    if set(args.metrics) & {"pr_curve"}:  # compute for overall and per label
        reduced_n = min(args.reduced_n, rr.shape[0], rg.shape[0])

        logger.info(
            f"Computing PR curve with:\n"
            f"\t classifier =  {args.pr_curve_clf}\n"
            f"\t samples = {reduced_n}\n"
            f"\t k = {args.nearest_k}"
        )

        rng = np.random.default_rng(seed)
        inds0 = rng.choice(rr.shape[0], reduced_n, replace=False)
        inds1 = rng.choice(rg.shape[0], reduced_n, replace=False)

        precisions, recalls = compute_pr_curve(
            rr[inds0],
            rg[inds1],
            lambdas=LAMBDAS,
            clf=args.pr_curve_clf,
            k=args.nearest_k,
        )
        prc_dict = {
            "nreal": rr[inds0].shape[0],  # not modfified inside `compute_pr_curve`
            "nfake": rg[inds1].shape[0],  # not modfified inside `compute_pr_curve`
            "precisions": precisions,
            "recalls": recalls,
        }
        scores[f"pr_curve_{args.pr_curve_clf}"] = prc_dict

    if "sw_approx" in args.metrics and label_key == "overall":
        print("Aprroximating Sliced W2.", file=sys.stderr)
        scores["sw_approx"] = sw_approx(rr, rg)

    if "vendi" in args.metrics and label_key == "overall":
        print("Calculating diversity score", file=sys.stderr)
        # scores['vendi'] = compute_vendi_score(reps[1])
        vendi_scores = compute_per_class_vendi_scores(rg, labels[1])
        scores["mean_vendi_per_class"] = vendi_scores.mean()

    logger.debug(f"{label_key} scores:")
    logger.debug(pformat(scores))

    return scores, vendi_scores


####################################################################################################
# Aggregate
####################################################################################################


def aggregate_runs(values):
    """Aggregate one metric across runs into (mean, std).

    Handles plain scalars as well as nested dicts (e.g. per-label score dicts),
    and array-valued metrics (e.g. precision/recall curves).
    For array-valued metrics, aggregates per-point across runs.
    """
    if isinstance(values[0], dict):
        mean, std = {}, {}
        for k in values[0]:
            mean[k], std[k] = aggregate_runs([v[k] for v in values])
        return mean, std

    arr = np.array(values)
    # If arr is 2D+, it's an array-valued metric: aggregate per-point (along axis 0)
    # If arr is 1D, it's scalar values: aggregate globally
    if arr.ndim >= 2:
        return arr.mean(axis=0), arr.std(axis=0)
    else:
        return arr.mean(), arr.std()


####################################################################################################
# Labelwise
####################################################################################################


def labelwise_setup(args, labels):
    """Resolve per-label filtering inputs from the overall labels.

    Returns (real_labs, gen_labs, label_values). label_values is empty unless
    --per-label is set and both real and generated samples carry labels, in
    which case only the overall score is computed.
    """
    if not args.per_label or labels is None:
        return None, None, np.array([])

    real_labs = None if labels[0] is None else np.asarray(labels[0])
    gen_labs = None if labels[1] is None else np.asarray(labels[1])

    if (
        real_labs is None
        or gen_labs is None
        or len(real_labs) == 0
        or len(gen_labs) == 0
    ):
        return real_labs, gen_labs, np.array([])

    return real_labs, gen_labs, np.unique(real_labs)


def unpack_nested_labelwise_metric(scores, metric_name, run_scores):
    """Unpack metric with per-label results from scores into run_scores.

    Converts a metric dict that contains both overall and per-label results
    (keyed by "label-i") into the run_scores structure where each label_key
    gets metric entries with the metric_name prefix.

    Args:
        scores: dict containing metric_name with overall + per-label results
        metric_name: str, key in scores (e.g., "prdc")
        run_scores: dict to update with unpacked results
    """
    if metric_name not in scores:
        return

    metric_data = scores.pop(metric_name)
    label_keys = [k for k in metric_data.keys() if k.startswith("label-")]

    # Unpack per-label results
    for label_key in label_keys:
        label_data = metric_data.pop(label_key)
        run_scores[label_key] = {f"{metric_name}-{k}": v for k, v in label_data.items()}

    # Merge overall results into scores with metric prefix
    scores.update({f"{metric_name}-{k}": v for k, v in metric_data.items()})


def aggregate_labelwise_scores(run_scores):
    """Average label-* scores within a run into run_scores["agg"] (in place).

    Handles scalars, arrays, and nested dicts (e.g., pr_curve results).
    Also computes run_scores["diff"] as the difference between overall and agg.
    """
    logger.info("Aggregating labelwise scores into 'agg' entry")

    label_keys = [k for k in run_scores if k.startswith("label-")]
    if not label_keys:
        return

    sample_keys = [k for k in run_scores[label_keys[0]] if k != "realism"]
    run_scores["agg"] = {}
    run_scores["diff"] = {}

    for metric_name in sample_keys:
        metric_values = [run_scores[lk][metric_name] for lk in label_keys]
        mean, _ = aggregate_runs(metric_values)
        run_scores["agg"][metric_name] = mean
        # Compute diff as overall - agg
        overall_value = run_scores["overall"][metric_name]
        run_scores["diff"][metric_name] = overall_value - mean


####################################################################################################
# Run
####################################################################################################


def _run_plan(args, real_labs):
    """Yield ``(real_labs_r, sub_seed, tag)`` for each run.

    Each run advances exactly one source of randomness while pinning the other,
    so the across-run mean/std isolates that source:

    - ``--random-labels`` advances the *random permutation* (the real-label
      shuffle) each run while pinning the *random reduction* (the reduced_n
      subsampling seed), isolating label-assignment variance.
    - otherwise the true labels are kept and the *random reduction* is advanced
      each run, isolating reduction variance.

    ``sub_seed`` is the seed handed to ``compute_scores``; it drives the random
    reduction inside the prdc block.
    """
    for r in range(args.nruns):
        if args.random_labels:
            # advance the random permutation; pin the random reduction
            rng_perm = np.random.default_rng(args.seed + r)
            random_labs = None if real_labs is None else rng_perm.permutation(real_labs)
            yield random_labs, args.seed, "(random permutation)"
        else:
            # advance the random reduction; keep true labels
            yield real_labs, args.seed + r, ""


def _run_default(args, real_reps, fake_reps, test_reps, labels):
    """Standard per-label runs, with the optional ``--random-labels`` modifier.

    Per-label groups are obtained by filtering the overall reps with the real
    labels (true, or permuted under ``--random-labels``). The "overall" score is
    invariant to the random permutation, so when randomizing labels (random
    reduction pinned) it is computed once and reused across runs.
    """

    all_runs_scores = []  # list of per-run dicts: {label: scores_dict}
    vendi_scores = {}

    real_labs, fake_labs, label_values = labelwise_setup(args, labels)

    for r, (real_labs_, seed_, tag) in enumerate(_run_plan(args, real_labs)):
        print(f"\n=== Run {r + 1}/{args.nruns} {tag} ===")
        run_scores = defaultdict(dict)

        print("\n--- overall ---")
        print(f"samples with shapes {real_reps.shape} and {fake_reps.shape}\n")
        scores, vs = compute_scores(
            args,
            real_reps,
            fake_reps,
            test_reps,
            labels=[real_labs_, fake_labs],
            label_key="overall",
            seed=seed_,
        )
        if vs:
            vendi_scores = vs

        run_scores["overall"] = scores

        for idx, lab in enumerate(label_values):
            label_key = f"label-{idx}"
            rr = real_reps[real_labs_ == lab]
            rg = fake_reps[fake_labs == lab]

            print(f"\n--- {label_key} ---")
            print(f"samples with shapes {rr.shape} and {rg.shape}\n")

            scores, _ = compute_scores(
                args,
                rr,
                rg,
                test_reps,
                labels=None,
                label_key=label_key,
                seed=seed_,
            )
            run_scores[label_key].update(scores)

        aggregate_labelwise_scores(run_scores)
        all_runs_scores.append(run_scores)

        logger.debug("Run scores:")
        logger.debug(pformat(run_scores))

    return all_runs_scores, vendi_scores


def run_compute_score(args, real_reps, fake_reps, test_reps, labels=None):
    """reps_r, reps_g: overall real/generated representation arrays (Nimg, ndim).

    Per-label groups, when --per-label is set, are derived by filtering these
    overall reps with the labels -- no separate per-label representations needed.
    For knn_filter metric, all per-label results are computed in a single call.
    """
    all_runs_scores, vendi_scores = _run_default(
        args,
        real_reps,
        fake_reps,
        test_reps,
        labels,
    )

    # Compute mean (and std if nruns > 1) across runs for all labels
    all_scores = {}
    all_keys = all_runs_scores[0].keys()

    for key in all_keys:  # overall, label-i, agg
        all_metrics = all_runs_scores[0][key].keys()
        mean_scores, std_scores = {}, {}

        for metric in all_metrics:
            values = [run[key][metric] for run in all_runs_scores]
            if metric == "realism":
                mean_scores[metric] = values[-1]
                continue
            # values may be scalars or nested per-label dicts (knn_filter)
            mean_scores[metric], std_scores[metric] = aggregate_runs(values)

        all_scores[key] = mean_scores
        if args.nruns > 1:
            all_scores[f"{key}_std"] = std_scores

    logger.debug("Final scores:")
    logger.debug(pformat(all_scores))

    return all_scores, vendi_scores


####################################################################################################
# Save
####################################################################################################


def save_score(
    scores,
    output_dir,
    model,
    path,
    ckpt,
    nsample,
    is_only=False,
    metrics=None,
):
    print("\nSaving scores...", file=sys.stderr)

    ckpt_str = ""
    if ckpt is not None:
        ckpt_str = f"_ckpt-{os.path.splitext(os.path.basename(ckpt))[0]}"

    metrics_str = "" if metrics is None else "-" + "_".join(metrics)
    path_str = (
        "_".join([extend_path(p) for p in path]) if path is not None else "overall"
    )

    if is_only:
        out_str = f"inception_score-{path_str}{ckpt_str}_nimage-{nsample}.txt"
    else:
        out_str = f"{model}-{metrics_str}-{path_str}{ckpt_str}-nimage_{nsample}.txt"

    out_path = os.path.join(output_dir, out_str)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for key, value in scores.items():
            if key == "realism":
                continue
            if isinstance(value, dict):
                f.write(f"[{key}]\n")
                for k, v in value.items():
                    if k == "realism":
                        continue
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value} \n")

    print(f"Wrote scores to {out_path}\n", file=sys.stderr)


def save_scores(description, scores, args, is_only=False, vendi_scores={}):
    print("\nSaving scores from all generated paths...", file=sys.stderr)

    run_params = vars(args)
    run_params["reference_dataset"] = run_params["train"]
    run_params["generated_dataset"] = run_params["gen"]

    ckpt_str = ""
    logger.debug(pformat(scores))

    if is_only:
        description["scores"] = "is"

    out_str = make_str(description)

    results_dir = args.output_dir
    if args.xp in ["sweep_prdc_k"]:
        results_dir = os.path.join(results_dir, out_str.split("-k_")[0])
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    out_path = os.path.join(results_dir, out_str)
    fname = f"{out_path}.npz"

    np.savez(fname, scores=scores, run_params=run_params)
    print(f"Saved scores to {fname}\n", file=sys.stderr)

    if vendi_scores is not None and len(vendi_scores) > 0:
        description["scores"] = "vendi"
        df = pd.DataFrame.from_dict(data=vendi_scores)

        out_str = make_str(description)
        out_path = os.path.join(results_dir, out_str)
        print(f"Saving vendi score to {out_path}.csv", file=sys.stderr)
        df.to_csv(f"{out_path}.csv")
