import os
import sys
from pathlib import Path

import click
import numpy as np

from dgm_eval.dataloaders import get_dataset_name


def descriptive_statistics(path):
    stats = {"num_samples": 0, "num_labels": 0, "samples_per_label": []}

    subfolders = sorted(
        [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))],
        key=int,
    )

    if not subfolders:
        stats["num_samples"] = sum(
            1 for f in os.listdir(path) if f.endswith((".jpg", ".png"))
        )
        return stats

    stats["num_labels"] = len(subfolders)

    for folder in subfolders:
        folder_path = os.path.join(path, folder)
        count = sum(1 for f in os.listdir(folder_path) if f.endswith((".jpg", ".png")))
        stats["samples_per_label"].append(count)
        stats["num_samples"] += count

    return stats


####################################################################################################
####################################################################################################
####################################################################################################


@click.command()
@click.option("--paths", "-p",      help="Paths to dataset folders.",                                   type=str, multiple=True)  # fmt: skip
@click.option("--outdir", "-o",     help="Path to save the generated LaTeX tables.", metavar="DIR",     type=click.Path(), default="experiments")  # fmt: skip
def main(paths, outdir):
    output_dir = os.path.join(outdir, "descriptive-stats")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stats = {}
    for path in paths:
        ds_name = get_dataset_name(path)
        print(f"Descriptive statistics for {ds_name}:")
        ds_stats = descriptive_statistics(path)
        print(ds_stats)

        stats[ds_name] = ds_stats

        np.savez(os.path.join(output_dir, f"{ds_name}.npz"), **ds_stats)


if __name__ == "__main__":
    main()
