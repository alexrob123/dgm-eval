import logging
import os
import pathlib
import sys
from pprint import pformat

import click
import numpy as np
import pandas as pd
import torch

from dgm_eval.metrics import METRICS
from dgm_eval.metrics.pr_curve import PR_CURVE_CLFS

from .dataloaders import get_datamodule_from_path
from .heatmaps import visualize_heatmaps
from .infra import get_device_and_num_workers
from .metrics import compute_inception_score
from .models import MODELS, load_encoder
from .representations import compute_reps
from .samples import save_samples
from .scores import run_compute_score, save_score, save_scores

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)


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
        dataloaderi = get_datamodule_from_path(
            args.path[i],
            model_IS.transform,
            num_workers,
            args,
        )
        # dataloaderi.data_loader is a single overall loader
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


###################################################################################################
###################################################################################################
###################################################################################################


@click.command()
# Datasets
@click.option("--train",                type=str,                                               help="Paths to the images: real dataset.")  # fmt: skip
@click.option("--gen",                  type=str,   multiple=True,                              help="Paths to the images, generated dataset.")  # fmt: skip
@click.option("--train-dataset",        type=str,   default="imagenet",                         help="Dataset that model was trained on. Sets proper normalization for MAE.")  # fmt: skip
@click.option("--test-path",            type=str,   default=None,                               help="Path to test images")  # fmt: skip
@click.option("--nsample",              type=int,   default=50_000,                             help="Maximum number of images to use for calculation")  # fmt: skip
@click.option("--clean-resize",         is_flag=True, default=False,                            help="Use clean resizing (from pillow)")  # fmt: skip
@click.option("--desc-stats",           is_flag=True, default=False,                            help="Whether to compute descriptive statistics and save them in a latex table.")  # fmt: skip
# Encoders
@click.option("--model",                type=click.Choice(MODELS), default="dinov2",            help="Model to use for generating feature representations.")  # fmt: skip
@click.option("--arch",                 type=str,   default=None,                               help="Model architecture. If none specified use default specified in model class")  # fmt: skip
@click.option("--checkpoint",           type=click.Path(), default=None,                        help="Path of model checkpoint.")  # fmt: skip
@click.option("--load/--no-load",       default=True,                                           help="Load representations and statistics from previous runs if possible")  # fmt: skip
@click.option("--depth",                type=int,   default=0,                                  help="Negative depth for internal layers, positive 1 for after projection head.")  # fmt: skip
@click.option("--save",                 is_flag=True, default=False,                            help="Save representations to reps_dir for faster computation next time")  # fmt: skip
# Metrics
@click.option("--metrics",              type=str,   multiple=True, default=METRICS,             help="Metrics to compute")  # fmt: skip
@click.option("--splits",               type=int,   default=10,                                 help="Number of splits for Inception Score(is)")  # fmt: skip
@click.option("--nearest-k",            type=int,   default=None,                               help="Number of neighbours for PRDC. If None, set to sqrt of number of samples used for calculation.")  # fmt: skip
@click.option("--reduced-n",            type=int,   default=10_000,                             help="Number of samples used for train, baseline, test, and generated sets for FLS")  # fmt: skip
@click.option("--pr-curve-clf",         type=click.Choice(PR_CURVE_CLFS), default="knn",        help="Classifier for PR curve")  # fmt: skip
# Heatmaps
@click.option("--heatmaps",             is_flag=True, default=False,                            help="Generate heatmaps showing the fd focus on images.")  # fmt: skip
@click.option("--heatmaps-perturbation", is_flag=True, default=False,                           help="Add some perturbation to the images on which gradcam is applied.")  # fmt: skip
# Setup
@click.option("--xp",                   type=click.Choice(["sweep_prdc_k"]), default=None,      help="Experiment to run.")  # fmt: skip
@click.option("--nruns",                type=int,   default=1,                                  help="Number of runs to average scores over.")  # fmt: skip
@click.option("--per-label",            is_flag=True, default=False,                            help="Whether to compute metrics per label. Only implemented for prdc and vendi currently.")  # fmt: skip
@click.option("--random-labels",        is_flag=True, default=False,                            help="Replace the real labels with a fresh random permutation, varied across runs while the subsampling is held fixed.")  # fmt: skip
# Hardware
@click.option("--device",               type=str,   default=None,                               help="Device to use. Like cuda, cuda:0 or cpu")  # fmt: skip
@click.option("--num-workers",          type=int,   default=None,                               help="Number of processes to use for data loading. Defaults to `min(8, num_cpus)`")  # fmt: skip
@click.option("--batch-size",           type=int,   default=50,                                 help="Batch size to use")  # fmt: skip
@click.option("--seed",                 type=int,   default=0,                                  help="Random seed")  # fmt: skip
# Output
@click.option("--save-imgs",            is_flag=True, default=False,                            help="Saves sample images per dataset.")  # fmt: skip
@click.option("--output-dir",           type=click.Path(), default="out/",                      help="Directory to save outputs in")  # fmt: skip
def main(
    # Datasets
    train,
    gen,
    train_dataset,
    test_path,
    nsample,
    clean_resize,
    desc_stats,
    # Encoders
    model,
    arch,
    checkpoint,
    load,
    depth,
    save,
    # Metrics
    metrics,
    splits,
    nearest_k,
    reduced_n,
    pr_curve_clf,
    # Heatmaps
    heatmaps,
    heatmaps_perturbation,
    # Setup
    xp,
    nruns,
    per_label,
    random_labels,
    # Hardware
    device,
    num_workers,
    batch_size,
    seed,
    # Output
    save_imgs,
    output_dir,
):
    """Run evaluation on generated and real image datasets."""

    # Convert click parameters to a simple namespace-like object for compatibility with existing code
    class Args:
        pass

    args = Args()
    # Datasets
    args.train = train
    args.gen = list(gen) if gen else []
    args.train_dataset = train_dataset
    args.test_path = test_path
    args.nsample = nsample
    args.clean_resize = clean_resize
    args.desc_stats = desc_stats
    # Encoders
    args.model = model
    args.arch = arch
    args.checkpoint = checkpoint
    args.load = load
    args.depth = depth
    args.save = save
    # Metrics
    args.metrics = list(metrics) if metrics else list(METRICS)
    args.splits = splits
    args.nearest_k = nearest_k
    args.reduced_n = reduced_n
    args.pr_curve_clf = pr_curve_clf
    # Heatmaps
    args.heatmaps = heatmaps
    args.heatmaps_perturbation = heatmaps_perturbation
    # Setup
    args.xp = xp
    args.nruns = nruns
    args.per_label = per_label
    args.random_labels = random_labels
    # Hardware
    args.device = device
    args.num_workers = num_workers
    args.batch_size = batch_size
    args.seed = seed
    # Output
    args.save_imgs = save_imgs
    args.output_dir = output_dir

    run(args)


def run(args):
    logger.info(
        "\nRunning evaluation...\n"
        "------------------------------------------------------------"
        "\nArguments:\n"
        f"\t train: {args.train}\n"
        f"\t gen: {args.gen}\n"
        f"\t model: {args.model}\n"
        f"\t metrics: {sorted(args.metrics)}\n"
        f"\t nruns: {args.nruns}\n"
        f"\t experiment: {args.xp}\n"
    )

    device, num_workers = get_device_and_num_workers(args.device, args.num_workers)

    # --- EXPERIMENTS ---

    if args.xp is not None:
        args.output_dir = os.path.join(args.output_dir, args.xp)

    if args.xp == "sweep_prdc_k":
        if not set(args.metrics) & {"knn_filter", "prdc"}:
            raise ValueError("XP 'sweep_prdc_k'  requires relevant metrics")

    if "knn_filter" in args.metrics:
        if not args.per_label:
            logger.warning("Enabling --per-label for knn_filter metric")
            args.per_label = True

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

    # --- TRAIN REPRESENTATIONS ---

    real_dm = get_datamodule_from_path(
        args.train,
        model.transform,
        num_workers,
        args,
    )
    real_reps = compute_reps(
        real_dm,
        model,
        device,
        args,
    )

    if args.per_label and real_dm.labels is None:
        raise ValueError(f"No labels in real dataset {real_dm.dataset_name}")
    if args.save_imgs:
        save_samples("./out-data", real_dm)

    # --- TEST REPRESENTATIONS ---

    if args.test_path is not None:
        test_dm = get_datamodule_from_path(
            args.test_path,
            model.transform,
            num_workers,
            args,
        )
        test_reps = compute_reps(
            test_dm,
            model,
            device,
            args,
        )
    else:
        test_reps = None

    # --- GENERATED REPRESENTATIONS AND SCORES ---

    all_scores = {}
    vendi_scores = {}
    gen_dataset_names = []

    for i, path in enumerate(args.gen):
        # Get representations
        gen_dm_i = get_datamodule_from_path(
            path,
            model.transform,
            num_workers,
            args,
            sample_w_replacement=True if "train" in path else False,
        )
        gen_reps_i = compute_reps(
            gen_dm_i,
            model,
            device,
            args,
        )

        if args.per_label and gen_dm_i.labels is None:
            raise ValueError(f"No labels in generated dataset {gen_dm_i.dataset_name}")
        if args.save_imgs:
            save_samples("./out-data", gen_dm_i)

        gen_dataset_names.append(gen_dm_i.dataset_name)

        # Compute scores
        print(f"\nComputing scores between ref dataset and {path}\n")

        labels = [real_dm.labels, gen_dm_i.labels]
        scores_i, vendi_scores_i = run_compute_score(
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
            [real_dm.dataset_name, gen_dm_i.dataset_name],
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
                f"{args.model}_{real_dm.dataset_name}_{gen_dm_i.dataset_name}"
                + f"{'_perturbation' if args.heatmaps_perturbation else ''}_{args.seed}"
            )
            visualize_heatmaps(
                real_reps,
                gen_reps_i,
                model,
                dataset=gen_dm_i.data_set,
                results_dir=os.path.join(args.output_dir, "results"),
                results_suffix=heatmap_suffix,
                dataset_name=gen_dm_i.dataset_name,
                device=device,
                perturbation=args.heatmaps_perturbation,
                random_seed=args.seed,
            )

    # --- SAVE SCORES ---

    desc_model = args.model + "_" + args.arch if args.arch is not None else args.model
    desc_metrics = "+".join(sorted(args.metrics))
    if "pr_curve" in args.metrics:
        desc_metrics = desc_metrics.replace("pr_curve", f"pr_curve_{args.pr_curve_clf}")

    desc = {
        "train_ds": real_dm.dataset_name,
        "gen_ds": "_".join(gen_dataset_names),
        "model": desc_model,
        "metrics": desc_metrics,
        "nimgs": len(real_dm.dataloader.dataset),
    }

    if args.nruns > 1:
        desc["nruns"] = args.nruns

    if set(args.metrics) & {"fls", "fls_overfit", "knn_filter", "prdc", "pr_curve"}:
        if args.reduced_n != args.nsample:
            desc["reduced"] = args.reduced_n

    if set(args.metrics) & {"knn_filter", "prdc", "pr_curve"}:
        desc["k"] = args.nearest_k

    if args.random_labels:
        desc["random_labs"] = ""

    logger.debug("Description for saving scores:")
    logger.debug(pformat(desc))

    save_scores(
        desc,
        all_scores,
        args,
        vendi_scores=vendi_scores,
    )


if __name__ == "__main__":
    main()
