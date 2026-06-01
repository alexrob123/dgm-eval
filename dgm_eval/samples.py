import os
import pathlib
import sys

import numpy as np
from torchvision.utils import save_image


def save_samples(output_dir, DL, n=8):
    samples_dir = os.path.join(output_dir, DL.dataset_name, "samples")
    pathlib.Path(samples_dir).mkdir(parents=True, exist_ok=True)

    label = "overall"

    existing = [f for f in os.listdir(samples_dir) if f.startswith(f"{label}_")]
    n_missing = n - len(existing)

    if n_missing <= 0:
        print(f"Samples for {label} dist. exist, moving on.", file=sys.stderr)
        return

    print(f"Completing samples for {label} dist.", file=sys.stderr)

    batch = next(iter(DL.data_loader))
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    indices = np.random.choice(
        len(images), min(n_missing, len(images)), replace=False
    )

    start = len(existing)
    for offset, idx in enumerate(indices):
        save_image(
            images[int(idx)],
            os.path.join(samples_dir, f"{label}_{start + offset}.png"),
            normalize=True,
        )
