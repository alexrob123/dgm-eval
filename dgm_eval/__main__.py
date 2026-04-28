import os
import pathlib
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
import torch

from .dataloaders import get_dataloader, get_dataloader_from_path
from .heatmaps import visualize_heatmaps
from .infra import get_device_and_num_workers
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
    compute_inception_score,
    compute_mmd,
    compute_per_class_vendi_scores,
    compute_prdc,
    sw_approx,
)
from .models import MODELS, load_encoder
from .representations import get_reps, load_reps, save_reps
from .samples import save_samples

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

# Data args
parser.add_argument(
    "--train",
    type=str,
    help="Paths to the images: real dataset.",
)
parser.add_argument(
    "--gen",
    type=str,
    nargs="+",
    help="Paths to the images, generated dataset.",
)
parser.add_argument(
    "--train_dataset",
    type=str,
    default="imagenet",
    help="Dataset that model was trained on. Sets proper normalization for MAE.",
)
parser.add_argument(
    "--test_path",
    type=str,
    default=None,
    help=("Path to test images"),
)
parser.add_argument(
    "--nsample",
    type=int,
    default=50_000,
    help="Maximum number of images to use for calculation",
)
parser.add_argument(
    "--clean_resize",
    action="store_true",
    help="Use clean resizing (from pillow)",
)
# Metrics args
parser.add_argument(
    "--metrics",
    type=str,
    nargs="+",
    default=[
        "fd",
        "fd-infinity",
        "kd",
        "prdc",
        "is",
        "authpct",
        "ct",
        "ct_test",
        "ct_modified",
        "fls",
        "fls_overfit",
        "vendi",
        "sw_approx",
    ],
    help="metrics to compute",
)
parser.add_argument(
    "--per-label",
    action="store_true",
    help="Whether to compute metrics per label. Only implemented for prdc and vendi currently.",
)
parser.add_argument(
    "--splits",
    type=int,
    default=10,
    help="num of splits for Inception Score(is)",
)
parser.add_argument(
    "--nearest_k",
    type=int,
    default=5,
    help="Number of neighbours for precision, recall, density, and coverage",
)
parser.add_argument(
    "--reduced_n",
    type=int,
    default=10_000,
    help="Number of samples used for train, baseline, test, and generated sets for FLS",
)
# Model args
parser.add_argument(
    "--model",
    type=str,
    default="dinov2",
    choices=MODELS.keys(),
    help="Model to use for generating feature representations.",
)
parser.add_argument(
    "--arch",
    type=str,
    default=None,
    help="Model architecture. If none specified use default specified in model class",
)
parser.add_argument(
    "-ckpt",
    "--checkpoint",
    type=str,
    default=None,
    help="Path of model checkpoint.",
)
parser.add_argument(
    "--load",
    action="store_true",
    help="Load representations and statistics from previous runs if possible",
)
parser.set_defaults(load=True)
parser.add_argument(
    "--no-load",
    action="store_false",
    dest="load",
    help="Do not load representations and statistics from previous runs.",
)
parser.add_argument(
    "--depth",
    type=int,
    default=0,
    help="Negative depth for internal layers, positive 1 for after projection head.",
)
# DataLoader args
parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=50,
    help="Batch size to use",
)
parser.add_argument(
    "--num-workers",
    type=int,
    help="Number of processes to use for data loading. Defaults to `min(8, num_cpus)`",
)
# Heatmaps args
parser.add_argument(
    "--heatmaps",
    action="store_true",
    help="Generate heatmaps showing the fd focus on images.",
)
parser.add_argument(
    "--heatmaps-perturbation",
    action="store_true",
    help="Add some perturbation to the images on which gradcam is applied.",
)
# Randomness and device args
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device to use. Like cuda, cuda:0 or cpu",
)
parser.add_argument(
    "--seed",
    type=int,
    default=13579,
    help="Random seed",
)
# Output args
parser.add_argument(
    "--save",
    action="store_true",
    help="Save representations to output_dir",
)
parser.add_argument(
    "--save-imgs",
    action="store_true",
    dest="save_imgs",
    help="Saves sample images per dataset.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="experiments/",
    help="Directory to save outputs in",
)


# def extend_path(p):
#     base = os.path.basename(p)
#     parent = os.path.dirname(p)
#     parent_base = os.path.basename(parent)

#     if parent_base in ("train", "test"):
#         grandparent_base = os.path.basename(os.path.dirname(parent))
#         return grandparent_base + "-" + parent_base + "-" + base

#     if base.isdigit():
#         return parent_base + "-" + base

#     return base


def extend_path(p):
    parts = []
    current = p

    while True:
        base = os.path.basename(current)
        parent = os.path.dirname(current)

        parts.append(base)

        if base.isdigit() or base in ("train", "test"):
            current = parent
        else:
            break

    return "-".join(reversed(parts))


def compute_representations(DL, model, device, args):
    """
    Load representations from disk if path exists,
    else compute image representations using the specified encoder

    Returns:
        repsi: float32 (Nimage, ndim)
    """
    print(f"\nGetting representations for dataset: {DL.dataset_name}", file=sys.stderr)
    reps_dir = os.path.join(args.output_dir, "data", DL.dataset_name, "reps")

    reps = []
    for i, dl in enumerate(DL.data_loader):
        label = "overall" if i == 0 else f"label-{i - 1}"
        repsi = None

        if args.load:
            repsi = load_reps(reps_dir, args.model, None, dl, label=label)
            if repsi is not None:
                reps.append(repsi)
                continue

        if repsi is None:
            print("Calculating reps...", file=sys.stderr)
            repsi = get_reps(model, dl, device, normalized=False)
            reps.append(repsi)

            if args.save:
                print(f"Saving reps to {reps_dir}", file=sys.stderr)

                hparams = vars(DL).copy()
                # Remove keys that can't be pickled
                hparams.pop("transform")
                hparams.pop("data_loader")
                hparams.pop("data_set")

                save_reps(
                    reps_dir,
                    repsi,
                    args.model,
                    None,
                    dl,
                    label=label,
                    hparams=hparams,
                )

    return reps


def compute_scores(args, reps_r, reps_g, test_reps, labels=None):
    """
    reps: list of two arrays corresponding to real reps and gen reps, each of shape (Nimage, ndim)
    labels: array of shape (Nimage,) with class labels for generated samples, if available
    """

    assert len(reps_r) == len(reps_g), "Num of real reps and real labels must match"

    all_scores = {}
    vendi_scores = None

    for i, (rr, rg) in enumerate(zip(reps_r, reps_g)):
        label = "overall" if i == 0 else f"label-{i - 1}"
        print(f"\n--- {label} ---", file=sys.stderr)
        print(
            f"samples with shapes {rr.shape} and {rg.shape}\n",
            file=sys.stderr,
        )
        scores = {}

        if "fd" in args.metrics and i == 0:
            print("Computing FD \n", file=sys.stderr)
            scores["fd"] = compute_FD_with_reps(rr, rg)

        if "fd_eff" in args.metrics and i == 0:
            print("Computing Efficient FD \n", file=sys.stderr)
            scores["fd_eff"] = compute_efficient_FD_with_reps(rr, rg)

        if "fd-infinity" in args.metrics and i == 0:
            print("Computing fd-infinity \n", file=sys.stderr)
            scores["fd_infinity_value"] = compute_FD_infinity(rr, rg)

        if "kd" in args.metrics and i == 0:
            print("Computing KD \n", file=sys.stderr)
            mmd_values = compute_mmd(rr, rg)
            scores["kd_value"] = mmd_values.mean()
            scores["kd_variance"] = mmd_values.std()

        if "prdc" in args.metrics:  # compute for all labels
            print("Computing PRDC", file=sys.stderr)

            reduced_n = min(args.reduced_n, rr.shape[0], rg.shape[0])
            inds0 = np.random.choice(rr.shape[0], reduced_n, replace=False)

            inds1 = np.arange(rg.shape[0])
            if "realism" not in args.metrics:
                # Realism is returned for each sample, so do not shuffle if this metric is desired.
                # Else filenames and realism scores will not align
                inds1 = np.random.choice(
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

        if "vendi" in args.metrics and i == 0:
            print("Calculating diversity score", file=sys.stderr)
            # scores['vendi'] = compute_vendi_score(reps[1])
            vendi_scores = compute_per_class_vendi_scores(rg, labels[1])
            scores["mean vendi per class"] = vendi_scores.mean()

        if "authpct" in args.metrics and i == 0:
            print("Computing authpct \n", file=sys.stderr)
            scores["authpct"] = compute_authpct(rr, rg)

        if "sw_approx" in args.metrics and i == 0:
            print("Aprroximating Sliced W2.", file=sys.stderr)
            scores["sw_approx"] = sw_approx(rr, rg)

        if "ct" in args.metrics and i == 0:
            print("Computing ct score \n", file=sys.stderr)
            scores["ct"] = compute_CTscore(rr, test_reps, rg)

        if "ct_test" in args.metrics and i == 0:
            print(
                "Computing ct score, modified to identify mode collapse only \n",
                file=sys.stderr,
            )
            scores["ct_test"] = compute_CTscore_mode(rr, test_reps, rg)

        if "ct_modified" in args.metrics and i == 0:
            print(
                "Computing ct score, modified to identify memorization only \n",
                file=sys.stderr,
            )
            scores["ct_modified"] = compute_CTscore_mem(rr, test_reps, rg)

        if "fls" in args.metrics or "fls_overfit" in args.metrics and i == 0:
            train_reps, gen_reps = rr, rg
            reduced_n = min(
                args.reduced_n,
                train_reps.shape[0] // 2,
                test_reps.shape[0],
                gen_reps.shape[0],
            )

            test_reps = test_reps[
                np.random.choice(test_reps.shape[0], reduced_n, replace=False)
            ]
            gen_reps = gen_reps[
                np.random.choice(gen_reps.shape[0], reduced_n, replace=False)
            ]

            print("Computing fls \n", file=sys.stderr)
            # fls must be after ot, as it changes train_reps
            train_reps = train_reps[
                np.random.choice(train_reps.shape[0], 2 * reduced_n, replace=False)
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

        for key, value in scores.items():
            if key != "realism":
                print(f"{key}: {value:.5f}", file=sys.stderr)

        all_scores[label] = scores

    return all_scores, vendi_scores


def save_score(scores, output_dir, model, path, ckpt, nsample, is_only=False):

    ckpt_str = ""
    if ckpt is not None:
        ckpt_str = f"_ckpt-{os.path.splitext(os.path.basename(ckpt))[0]}"

    if is_only:
        out_str = f"Inception_score_{'-'.join([extend_path(p) for p in path])}{ckpt_str}_nimage-{nsample}.txt"
    else:
        out_str = f"fd_{model}_{'-'.join([extend_path(p) for p in path])}{ckpt_str}_nimage-{nsample}.txt"

    out_path = os.path.join(output_dir, out_str)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for key, value in scores.items():
            if key == "realism":
                continue
            f.write(f"{key}: {value} \n")


def save_scores(scores, args, is_only=False, vendi_scores={}):

    results_dir = os.path.join(args.output_dir, "results")
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    run_params = vars(args)
    run_params["reference_dataset"] = run_params["train"]
    run_params["generated_dataset"] = run_params["gen"]

    ckpt_str = ""
    print(scores, file=sys.stderr)

    if is_only:
        out_str = f"Inception_scores_nimage-{args.nsample}"
    else:
        out_str = f"{args.model}{ckpt_str}_scores_nimage-{args.nsample}"
    out_path = os.path.join(results_dir, out_str)

    fname = f"{out_path}.npz"
    print(f"Saving scores to {fname}\n", file=sys.stderr)
    np.savez(fname, scores=scores, run_params=run_params)

    if vendi_scores is not None and len(vendi_scores) > 0:
        df = pd.DataFrame.from_dict(data=vendi_scores)
        out_str = f"{args.model}{ckpt_str}_vendi_scores_nimage-{args.nsample}"
        out_path = os.path.join(results_dir, out_str)
        print(f"saving vendi score to {out_path}.csv", file=sys.stderr)
        df.to_csv(f"{out_path}.csv")


def get_inception_scores(args, device, num_workers):
    # The inceptionV3 with logit output is only used for calculate inception score
    print(
        "Computing Inception score with model = inception, ckpt=None, and dims=1008.",
        file=sys.stderr,
    )
    print("Loading Model", file=sys.stderr)

    IS_scores = {}
    model_IS = load_encoder(
        "inception",
        device,
        ckpt=None,
        dims=1008,
        arch=None,
        pretrained_weights=None,
        train_dataset=None,
        clean_resize=args.clean_resize,
        depth=args.depth,
    )

    for i, path in enumerate(args.path[1:]):
        print(f"\nGetting DataLoader for path: {path}\n", file=sys.stderr)
        dataloaderi = get_dataloader_from_path(
            args.path[i],
            model_IS.transform,
            num_workers,
            args,
        )
        # dataloaderi.data_loader is a list
        print(f"Computing inception score for {path}\n", file=sys.stderr)
        IS_score_i = compute_inception_score(
            model_IS,
            DataLoader=dataloaderi,
            splits=args.splits,
            device=device,
        )
        IS_scores[f"run{i:02d}"] = IS_score_i
        print(IS_score_i)
    save_scores(IS_scores, args, is_only=True)
    if len(args.metrics) == 1:
        sys.exit(0)

    return IS_scores


def extend_paths_with_subdirs(paths):
    print("Extending paths to subdirs for metrics per label.")

    extended_paths = [paths[0]]
    for path_i in paths[1:]:
        extended_paths.append(path_i)
        subdirs = [os.path.join(path_i, subdir) for subdir in os.listdir(path_i)]
        if subdirs:
            extended_paths.extend(subdirs)
    print(f"New paths: {extended_paths}")

    return extended_paths


# def get_dl_and_compute_reps(path, model, device, args, sample_w_replacement=False):
#     dataloader = get_dataloader_from_path(
#         path,
#         model.transform,
#         args.num_workers,
#         args,
#         sample_w_replacement=sample_w_replacement,
#     )
#     repsi = compute_representations(dataloader, model, device, args)
#     return dataloader, repsi


###################################################################################################
###################################################################################################
###################################################################################################


def run(args):
    print(
        "\nRunning evaluation...\n"
        "------------------------------------------------------------"
        "\nArguments:\n"
        f"\ttrain: {args.train}\n"
        f"\tgen: {args.gen}\n"
        f"\tmodel: {args.model}\n"
    )

    device, num_workers = get_device_and_num_workers(args.device, args.num_workers)

    # IS does not require a reference dataset, so compute first.
    # IS_scores = None
    # if "is" in args.metrics and args.model == "inception":
    #     IS_scores = get_inception_scores(args, device, num_workers)

    # Move IS encoder here ?

    # Encoder
    model = load_encoder(
        args.model,
        device,
        ckpt=None,
        arch=None,
        clean_resize=args.clean_resize,
        sinception=True if args.model == "sinception" else False,
        depth=args.depth,
    )

    # Train representations
    dataloader_real = get_dataloader_from_path(
        args.train,
        model.transform,
        num_workers,
        args,
    )  # list
    reps_real = compute_representations(
        dataloader_real,
        model,
        device,
        args,
    )  # list
    labs_real = dataloader_real.labels

    if args.save_imgs:
        save_samples(args.output_dir, dataloader_real)

    # Test representations
    repsi_test = None
    if args.test_path is not None:
        dataloader_test = get_dataloader_from_path(
            args.test_path,
            model.transform,
            num_workers,
            args,
        )  # list
        repsi_test = compute_representations(
            dataloader_test,
            model,
            device,
            args,
        )  # list

    # Loop over all generated paths
    all_scores = {}
    vendi_scores = {}

    for i, path in enumerate(args.gen):
        dataloader_i = get_dataloader_from_path(
            path,
            model.transform,
            num_workers,
            args,
            sample_w_replacement=True if "train" in path else False,
        )  # list
        reps_i = compute_representations(
            dataloader_i,
            model,
            device,
            args,
        )  # list
        labs_i = dataloader_i.labels

        if args.save_imgs:
            save_samples(args.output_dir, dataloader_i)

        labels = [labs_real, labs_i]

        print(f"\nComputing scores between ref dataset and {path}\n", file=sys.stderr)
        scores_i, vendi_scores_i = compute_scores(
            args,
            reps_real,
            reps_i,
            repsi_test,
            labels=labels,
        )
        if vendi_scores_i is not None:
            vendi_scores[os.path.basename(path)] = vendi_scores_i

        print("\nSaving scores...", file=sys.stderr)
        save_score(
            scores_i,
            os.path.join(args.output_dir, "results"),
            args.model,
            [args.train, path],
            None,
            args.nsample,
        )
        # if IS_scores is not None:
        #     scores_i.update(IS_scores[f"run{i:02d}"])
        all_scores[f"run{i:02d}"] = scores_i

        if args.heatmaps:
            print("Visualizing FD gradient with gradcam\n", file=sys.stderr)
            heatmap_suffix = (
                f"{args.model}_{dataloader_real.dataset_name}_{dataloader_i.dataset_name}"
                + f"{'_perturbation' if args.heatmaps_perturbation else ''}_{args.seed}"
            )
            visualize_heatmaps(
                reps_real,
                reps_i,
                model,
                dataset=dataloader_i.data_set,
                results_dir=os.path.join(args.output_dir, "results"),
                results_suffix=heatmap_suffix,
                dataset_name=dataloader_i.dataset_name,
                device=device,
                perturbation=args.heatmaps_perturbation,
                random_seed=args.seed,
            )
    # save scores from all generated paths
    save_scores(all_scores, args, vendi_scores=vendi_scores)


def main():
    args = parser.parse_args()

    run(args)

    # if args.per_label:
    #     real_path = args.path[0]
    #     real_subdirs = sorted(os.listdir(real_path))
    #     real_labels = len(real_subdirs)
    #     real_label_paths = [os.path.join(real_path, d) for d in real_subdirs]

    #     gen_paths = args.path[1:]
    #     ext_gen_paths = []
    #     for path_i in gen_paths:
    #         gen_subdirs = sorted(os.listdir(path_i))
    #         gen_labels = len(gen_subdirs)

    #         assert real_labels == gen_labels, (
    #             f"Number of labels in real and generated datasets must match for per-label metrics. "
    #             f"Found {real_labels} and {gen_labels} for {real_path} and {path_i} respectively."
    #         )
    #         ext_gen_paths.append([os.path.join(path_i, d) for d in gen_subdirs])

    #     for i, real_p in enumerate(real_label_paths):
    #         gen_p = [ext_gen_paths[j][i] for j in range(len(gen_paths))]
    #         args.path = [real_p] + gen_p
    #         run(args)


if __name__ == "__main__":
    main()
