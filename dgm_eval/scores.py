import os
import pathlib
import sys

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
    compute_prdc,
    sw_approx,
)
from .utils import extend_path, make_str

####################################################################################################
# Compute
####################################################################################################


def compute_scores(
    args,
    rr,
    rg,
    test_reps,
    labels=None,
    label_name="overall",
    seed=0,
):
    """
    reps: list of two arrays corresponding to real reps and gen reps, each of shape (Nimage, ndim)
    labels: array of shape (Nimage,) with class labels for generated samples, if available
    """

    scores = {}
    vendi_scores = {}

    if "fd" in args.metrics and label_name == "overall":
        print("Computing FD \n", file=sys.stderr)
        scores["fd"] = compute_FD_with_reps(rr, rg)

    if "fd-eff" in args.metrics and label_name == "overall":
        print("Computing Efficient FD \n", file=sys.stderr)
        scores["fd_eff"] = compute_efficient_FD_with_reps(rr, rg)

    if "fd-infinity" in args.metrics and label_name == "overall":
        print("Computing fd-infinity \n", file=sys.stderr)
        scores["fd_infinity_value"] = compute_FD_infinity(rr, rg)

    if "kd" in args.metrics and label_name == "overall":
        print("Computing KD \n", file=sys.stderr)
        mmd_values = compute_mmd(rr, rg)
        scores["kd_value"] = mmd_values.mean()
        scores["kd_variance"] = mmd_values.std()

    if "prdc" in args.metrics:  # compute for overall and per label
        print("Computing PRDC", file=sys.stderr)

        rng = np.random.default_rng(seed)
        reduced_n = min(args.reduced_n, rr.shape[0], rg.shape[0])
        inds0 = rng.choice(rr.shape[0], reduced_n, replace=False)
        inds1 = np.arange(rg.shape[0])
        if "realism" not in args.metrics:
            # Realism is returned for each sample, so do not shuffle if this metric is desired.
            # Else filenames and realism scores will not align
            inds1 = rng.choice(
                inds1,
                min(inds1.shape[0], reduced_n),
                replace=False,
            )

        prdc_dict = compute_prdc(
            rr[inds0],
            rg[inds1],
            nearest_k=args.nearest_k,
            realism=True if "realism" in args.metrics else False,
        )
        scores.update(prdc_dict)

    if (
        "prdc" in args.metrics
        and args.xp == "knn-balls-filtering"
        and label_name == "overall"
    ):
        print("Computing PRDC per label with kNN balls filtering")

        rng = np.random.default_rng(seed)
        reduced_n = min(args.reduced_n, rr.shape[0], rg.shape[0])
        inds0 = rng.choice(rr.shape[0], reduced_n, replace=False)
        inds1 = rng.choice(rg.shape[0], reduced_n, replace=False)

        from .experiments import xp_knn_balls_filtering

        d = xp_knn_balls_filtering(
            rr[inds0],
            rg[inds1],
            labels[0][inds0],
            labels[1][inds1],
            nlabels=max(np.max(labels[0]), np.max(labels[1])) + 1,
            nearest_k=args.nearest_k,
        )

        return d, vendi_scores

    if "denscov" in args.metrics:
        raise NotImplementedError("denscov metric is currently not implemented")

    if "vendi" in args.metrics and label_name == "overall":
        print("Calculating diversity score", file=sys.stderr)
        # scores['vendi'] = compute_vendi_score(reps[1])
        vendi_scores = compute_per_class_vendi_scores(rg, labels[1])
        scores["mean vendi per class"] = vendi_scores.mean()

    if "authpct" in args.metrics and label_name == "overall":
        print("Computing authpct \n", file=sys.stderr)
        scores["authpct"] = compute_authpct(rr, rg)

    if "sw-approx" in args.metrics and label_name == "overall":
        print("Aprroximating Sliced W2.", file=sys.stderr)
        scores["sw_approx"] = sw_approx(rr, rg)

    if "ct" in args.metrics and label_name == "overall":
        print("Computing ct score \n", file=sys.stderr)
        scores["ct"] = compute_CTscore(rr, test_reps, rg)

    if "ct-test" in args.metrics and label_name == "overall":
        print(
            "Computing ct score, modified to identify mode collapse only \n",
            file=sys.stderr,
        )
        scores["ct_test"] = compute_CTscore_mode(rr, test_reps, rg)

    if "ct-modified" in args.metrics and label_name == "overall":
        print(
            "Computing ct score, modified to identify memorization only \n",
            file=sys.stderr,
        )
        scores["ct_modified"] = compute_CTscore_mem(rr, test_reps, rg)

    if (
        "fls" in args.metrics or "fls-overfit" in args.metrics
    ) and label_name == "overall":
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
        if "fls-overfit" in args.metrics:
            scores["fls-overfit"] = compute_fls_overfit(
                train_reps,
                baseline_reps,
                test_reps,
                gen_reps,
            )

    for key, value in scores.items():
        if key != "realism":
            print(f"{key}: {value:.5f}", file=sys.stderr)
    print("\n")

    return scores, vendi_scores


def compute_scores_wrapper(args, reps_r, reps_g, test_reps, labels=None):
    """
    reps: list of two arrays corresponding to real reps and gen reps, each of shape (Nimage, ndim)
    labels: array of shape (Nimage,) with class labels for generated samples, if available
    """

    assert len(reps_r) == len(reps_g), "Num of real reps and real labels must match"

    all_scores = {}
    vendi_scores = {}

    for i, (rr, rg) in enumerate(zip(reps_r, reps_g)):
        label = "overall" if i == 0 else f"label-{i - 1}"

        print(f"\n--- {label} ---")
        print(f"samples with shapes {rr.shape} and {rg.shape}\n")

        runs_scores = []

        for r in range(args.nruns):
            print(f">> Run {r + 1}/{args.nruns}")

            scores, vs = compute_scores(
                args,
                rr,
                rg,
                test_reps,
                labels=labels,
                label_name=label,
                seed=args.seed + r,
            )

            if vs:
                vendi_scores = vs

            runs_scores.append(scores)

        if args.nruns == 1:
            all_scores[label] = runs_scores[0]

        else:
            # args.nruns > 1
            # process scores from all runs to compute mean and std
            mean_scores, std_scores = {}, {}

            keys = runs_scores[0].keys()
            for key in keys:
                values = [run_score[key] for run_score in runs_scores]
                if key == "realism":
                    # per-sample array; keep last run's value, no std
                    mean_scores[key] = values[-1]
                    continue
                values_array = np.array(values)
                mean_scores[key] = values_array.mean()
                std_scores[key] = values_array.std()

            all_scores[label] = mean_scores
            all_scores[f"{label}_std"] = std_scores

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

    metrics_str = "" if metrics is None else "_" + "-".join(metrics)
    path_str = (
        "-".join([extend_path(p) for p in path]) if path is not None else "overall"
    )

    if is_only:
        out_str = f"inception_score_{path_str}{ckpt_str}_nimage-{nsample}.txt"
    else:
        out_str = (
            f"{model}_scores-{metrics_str}_{path_str}{ckpt_str}_nimage-{nsample}.txt"
        )

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
    print(scores, file=sys.stderr)

    if is_only:
        description["scores"] = "is"

    out_str = make_str(description)

    results_dir = args.output_dir
    if args.xp in ("sweep-prdc-k", "knn-balls-filtering"):
        results_dir = os.path.join(results_dir, out_str.split("_k-")[0])
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
        print(f"saving vendi score to {out_path}.csv", file=sys.stderr)
        df.to_csv(f"{out_path}.csv")
