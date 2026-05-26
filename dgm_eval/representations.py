import os
import pathlib
import sys

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

####################################################################################################
# Compute
####################################################################################################


def compute_reps(DL, model, device, args):
    """
    Load representations from disk if path exists,
    else compute image representations using the specified encoder

    Returns:
        repsi: float32 (Nimage, ndim)
    """
    print(f"\nGetting representations for dataset: {DL.dataset_name}", file=sys.stderr)
    reps_dir = os.path.join(
        "./out-data",
        DL.dataset_name,
        "reps",
        args.xp if args.xp in ["random-labels"] else "",
    )

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


def get_reps(model, dataloader, device, normalized=False):
    """Extracts features from all images in DataLoader given model.

    Params:
    -- model       : Instance of Encoder such as inception or CLIP or dinov2
    -- dataloader  : torch.dataloader containing image files, or torchvision.dataset

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    start_idx = 0
    for ibatch, batch in enumerate(tqdm(dataloader)):
        if isinstance(batch, list):
            # batch is likely list[array(images), array(labels)]
            batch = batch[0]

        if not torch.is_tensor(batch):
            # assume batch is then e.g. AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
            batch = batch["pixel_values"]
            batch = batch[:, 0]

        # Convert grayscale to RGB
        if batch.ndim == 3:
            batch.unsqueeze_(1)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

            if not torch.is_tensor(pred):  # Some encoders output tuples or lists
                pred = pred[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.dim() > 2:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)

        if normalized:
            pred = torch.nn.functional.normalize(pred, dim=-1)
        pred = pred.cpu().numpy()

        if ibatch == 0:
            # initialize output array with full dataset size
            dims = pred.shape[-1]
            pred_arr = np.empty((len(dataloader.dataset), dims))

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def load_reps(saved_dir, model, checkpoint, dataloader, label=None):
    """Save representations and other info to disk at file_path"""
    save_path = get_path(saved_dir, model, checkpoint, dataloader, label)
    reps = None
    print("Loading from:", save_path)
    if os.path.exists(f"{save_path}.npz"):
        saved_file = np.load(f"{save_path}.npz")
        reps = saved_file["reps"]
    return reps


####################################################################################################
# Save
####################################################################################################


def save_reps(output_dir, reps, model, checkpoint, dataloader, label, hparams):
    """Save representations and other info to disk at file_path"""
    out_path = get_path(output_dir, model, checkpoint, dataloader, label)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    np.savez(out_path, model=model, reps=reps, hparams=hparams)


def get_path(output_dir, model, checkpoint, dataloader, label=None):
    # train_str = "train" if DataLoader.train_set else "test"

    ckpt_str = (
        ""
        if checkpoint is None
        else f"_ckpt-{os.path.splitext(os.path.basename(checkpoint))[0]}"
    )

    hparams_str = f"reps_{label}_{model}{ckpt_str}_nimgs-{len(dataloader.dataset)}"
    # _{train_str}
    return os.path.join(output_dir, hparams_str)
