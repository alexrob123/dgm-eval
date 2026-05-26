import os
import pathlib
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
import torch

from .dataloaders import get_dataloader_from_path
from .heatmaps import visualize_heatmaps
from .infra import get_device_and_num_workers
from .metrics import compute_inception_score
from .models import MODELS, load_encoder
from .representations import compute_reps
from .samples import save_samples
from .scores import compute_scores_wrapper, save_score, save_scores


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


# def get_dl_reps_labs(path, args, model, device, num_workers, **kwargs):
#     """Gets dataloader, representations, and labels for dataset"""

#     dl = get_dataloader_from_path(
#         path,
#         model.transform,
#         num_workers,
#         args,
#         **kwargs,
#     )  # list

#     reps = compute_reps(
#         dl,
#         model,
#         device,
#         args,
#     )  # list

#     labs = dl.labels

#     return dl, reps, labs


###################################################################################################
###################################################################################################
###################################################################################################


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
    "--train-dataset",
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
        "denscov",  # under development
        "is",
        "authpct",
        "ct",
        "ct-test",
        "ct-modified",
        "fls",
        "fls-overfit",
        "vendi",
        "sw-approx",
    ],
    help="metrics to compute",
)
parser.add_argument(
    "--per-label",
    action="store_true",
    help="Whether to compute metrics per label. Only implemented for prdc and vendi currently.",
)
parser.add_argument(
    "--nruns",
    type=int,
    default=1,
    help="Number of runs to average scores over.",
)
parser.add_argument(
    "--splits",
    type=int,
    default=10,
    help="num of splits for Inception Score(is)",
)
parser.add_argument(
    "--nearest-k",
    type=int,
    default=None,
    help="Number of neighbours for PRDC. If None, set to sqrt of number of samples used for calculation.",
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
# Descriptive stats args
parser.add_argument(
    "--desc-stats",
    action="store_true",
    help="Whether to compute descriptive statistics and save them in a latex table.",
)
# Experiments args
parser.add_argument(
    "--xp",
    type=str,
    default=None,
    choices=["sweep-prdc-k", "random-labels", "knn-balls-filtering"],
    help="Experiment to run.",
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
    help="Save representations to reps_dir for faster computation next time",
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

    # --- EXPERIMENTS ---

    if args.xp is not None:
        print(f"Running experiment: {args.xp}\n")
        args.output_dir = os.path.join(args.output_dir, args.xp)

    if args.xp in ["sweep-prdc-k", "knn-balls-filtering"]:
        args.metrics = ["prdc"]

    # --- QUICK INCEPTION SCORE OPTION ---

    # IS does not require a reference dataset, so compute first.
    # IS_scores = None
    # if "is" in args.metrics and args.model == "inception":
    #     IS_scores = get_inception_scores(args, device, num_workers)

    # Move IS encoder here ?

    # --- ENCODER ---

    model = load_encoder(
        args.model,
        device,
        ckpt=None,
        arch=None,
        clean_resize=args.clean_resize,
        sinception=True if args.model == "sinception" else False,
        depth=args.depth,
    )

    descriptive_stats = {}

    # --- TRAIN REPRESENTATIONS ---

    real_dl = get_dataloader_from_path(
        args.train,
        model.transform,
        num_workers,
        args,
    )  # list
    real_reps = compute_reps(
        real_dl,
        model,
        device,
        args,
    )  # list
    real_labs = real_dl.labels

    if args.save_imgs:
        save_samples("./out-data", real_dl)

    # --- TEST REPRESENTATIONS ---

    if args.test_path is not None:
        test_dl = get_dataloader_from_path(
            args.test_path,
            model.transform,
            num_workers,
            args,
        )  # list
        test_reps = compute_reps(
            test_dl,
            model,
            device,
            args,
        )  # list
    else:
        test_reps = None

    # --- GENERATED REPRESENTATIONS AND SCORES ---

    all_scores = {}
    vendi_scores = {}
    gen_dataset_names = []

    for i, path in enumerate(args.gen):
        gen_dl_i = get_dataloader_from_path(
            path,
            model.transform,
            num_workers,
            args,
            sample_w_replacement=True if "train" in path else False,
        )  # list
        gen_reps_i = compute_reps(
            gen_dl_i,
            model,
            device,
            args,
        )  # list
        gen_labs_i = gen_dl_i.labels

        if args.save_imgs:
            save_samples("./out-data", gen_dl_i)

        gen_dataset_names.append(gen_dl_i.dataset_name)

        labels = [real_labs, gen_labs_i]

        print(f"\nComputing scores between ref dataset and {path}\n", file=sys.stderr)
        scores_i, vendi_scores_i = compute_scores_wrapper(
            args,
            real_reps,
            gen_reps_i,
            test_reps,
            labels=labels,
        )
        if vendi_scores_i:
            vendi_scores[os.path.basename(path)] = vendi_scores_i

        save_score(
            scores_i,
            args.output_dir,
            args.model,
            [real_dl.dataset_name, gen_dl_i.dataset_name],
            None,
            args.nsample,
            metrics=args.metrics,
        )
        # if IS_scores is not None:
        #     scores_i.update(IS_scores[f"run{i:02d}"])
        all_scores[f"run{i:02d}"] = scores_i

        if args.heatmaps:
            print("Visualizing FD gradient with gradcam\n", file=sys.stderr)
            heatmap_suffix = (
                f"{args.model}_{real_dl.dataset_name}_{gen_dl_i.dataset_name}"
                + f"{'_perturbation' if args.heatmaps_perturbation else ''}_{args.seed}"
            )
            visualize_heatmaps(
                real_reps,
                gen_reps_i,
                model,
                dataset=gen_dl_i.data_set,
                results_dir=os.path.join(args.output_dir, "results"),
                results_suffix=heatmap_suffix,
                dataset_name=gen_dl_i.dataset_name,
                device=device,
                perturbation=args.heatmaps_perturbation,
                random_seed=args.seed,
            )

    # --- SAVE SCORES ---

    desc = {
        "real_ds": real_dl.dataset_name,
        "gen_ds": "-".join(gen_dataset_names),
        "model": args.model + "-" + args.arch if args.arch is not None else args.model,
        "scores": "-".join(args.metrics),
        "nimgs": args.nsample
        if args.nsample > 0
        else len(real_dl.data_loader[0].dataset),
    }
    if "prdc" in args.metrics:
        desc["k"] = args.nearest_k
    if args.nruns > 1:
        desc["nruns"] = args.nruns

    save_scores(
        desc,
        all_scores,
        args,
        vendi_scores=vendi_scores,
    )


def main():
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
