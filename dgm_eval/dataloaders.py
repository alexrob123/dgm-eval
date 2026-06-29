import logging
import os
import pathlib
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS | {ext.upper() for ext in IMAGE_EXTENSIONS}

TORCHVISION_DATA_PATH = "./data/"


def get_files_at_path(path):
    """Return list of all files at path of type IMAGE_EXTENSIONS"""

    files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob(f"*.{ext}")])

    return files


def get_dataset_name(p):
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

    return "_".join(reversed(parts)).replace("-", "_")


class ImagePathDataset(torch.utils.data.Dataset):
    """
    Create a custom dataset from a list of image files on disk

    Files must have image extensions specified in IMAGE_EXTENSIONS
    """

    def __init__(self, files, transform=None):
        self.files = sorted(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class NpzDataset(torch.utils.data.Dataset):
    """
    Create a custom dataset from a npz file of images, as used in ADM's evaluation code.
    See https://github.com/openai/guided-diffusion/tree/main/evaluations for more details.
    """

    def __init__(self, path, transform=None):
        self.path = path
        self.data = np.load(path)["arr_0"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class DataModule:
    """
    Create Datasets and Dataloaders from ImagePathDataset and from torchvision.datasets.
    """

    def __init__(
        self,
        path,
        train_set=False,
        nsample=-1,
        transform=None,
        batch_size=50,
        num_workers=1,
        seed=0,
        random_sample=True,
        sample_w_replacement=False,
    ):

        self.path = path
        self.train_set = train_set
        self.nsample = nsample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # for class conditional models, remember the labels as loading
        self.labels = []

        self.random_sample = random_sample
        self.sample_w_replacement = sample_w_replacement

        if sample_w_replacement:
            print(
                (
                    f"Warning: sample_w_replacement={sample_w_replacement}."
                    f"Sampling with replacement from path {path}"
                ),
                file=sys.stderr,
            )
            self.seed += 1

        self.transform = transform
        if not transform:
            self.transform = torchvision.transforms.ToTensor()

        self.get_dataset()  # self.dataset, self.labels
        self.original_ds_len = len(self.dataset)

        if (self.nsample > 0) and (len(self.dataset) > self.nsample):
            self.subsample_dataset()  # self.dataset, self.labels
        self.ds_len = len(self.dataset)

        self.get_dataloader()

    def get_dataset(self):
        """
        Get dataset from local path or from torchvision.datasets

        Only `get_local_dataset` is triggering labels.
        """
        if os.path.exists(self.path):
            if os.path.isfile(self.path) and self.path.endswith(".npz"):
                self.get_local_adm_dataset()
            else:
                self.get_local_dataset()

        else:
            self.get_torchvision_dataset()

    def get_local_adm_dataset(self):
        """
        Get dataset stored in ADM npz format (see https://github.com/openai/guided-diffusion/tree/main/evaluations) from disk
        """

        self.dataset_name = os.path.splitext(
            os.path.basename(os.path.normpath(self.path))
        )[0]

        self.files = None
        self.labels = None
        # Confirm data at path is in proper format
        try:
            self.dataset = NpzDataset(self.path, transform=self.transform)
        except:
            raise RuntimeError(
                f"Images cannot be loaded from {self.path}. Expecting ADM-style npz file: {IMAGE_EXTENSIONS}"
            )

    def get_local_dataset(self):
        """
        Get dataset from disk

        Currently accepted formats:

        1.) Path to folder containing individual images of extension types in IMAGE_EXTENSIONS

        2.) Path to folder containing sub-folders for each image class,
            where each sub-folder contains individual images of extension types in IMAGE_EXTENSIONS
        """

        self.dataset_name = get_dataset_name(self.path)

        image_path = pathlib.Path(self.path)

        self.files = get_files_at_path(image_path)
        class_idx = 0

        def get_order(file):
            filename = os.path.splitext(os.path.basename(file))[0]
            return int(filename) if filename.isnumeric() else filename

        if not self.files:
            # Assume sub-folders for image classes
            class_dirs = sorted(
                image_path.glob("*"), key=get_order
            )  # look for all subfolders in the numerical order
            self.files = []
            for f in class_dirs:
                files_in_path = get_files_at_path(f)
                self.files += files_in_path
                self.labels.extend([class_idx for _ in range(len(files_in_path))])
                class_idx += 1
        self.labels = np.array(self.labels, dtype=np.int32)

        # Confirm data at path is in proper format
        try:
            self.dataset = ImagePathDataset(self.files, transform=self.transform)
        except:
            raise RuntimeError(
                f"Images cannot be loaded from {self.path}. Expecting path full of images: {IMAGE_EXTENSIONS}"
            )

    def get_torchvision_dataset(self):
        """Use torchvision.datasets"""
        print(f"Getting torchvision dataset: {self.path}", file=sys.stderr)

        self.dataset_name = self.path
        self.files = []  # empty list, as torchvision.datasets has various different formats
        try:
            torchvision_dataset = getattr(torchvision.datasets, self.dataset_name)

        except:
            raise RuntimeError(f"{self.dataset_name} is not a dataset in torchvision")

        else:
            self.dataset = torchvision_dataset(
                root=TORCHVISION_DATA_PATH,
                train=self.train_set,
                transform=self.transform,
                download=True,
            )

    def subsample_dataset(self):
        """subsample to desired size, respecting label prior if available"""
        logger.info(
            f"Subsampling dataset from {len(self.dataset)} to {self.nsample} samples, with "
            f"random_sample={self.random_sample}, sample_w_replacement={self.sample_w_replacement}"
        )

        rng = np.random.default_rng(self.seed)
        # for consistent subsampling of datasets across runs
        # local generator, no global state side effects

        if self.random_sample:
            # Use stratified sampling if labels are available
            if self.labels is not None and len(self.labels) > 0:
                self.inds_keep = self._stratified_subsample(rng)
            else:
                self.inds_keep = sorted(
                    rng.choice(
                        len(self.dataset),
                        self.nsample,
                        replace=self.sample_w_replacement,
                    )
                )
        else:
            self.inds_keep = np.arange(self.nsample)

        if self.files:
            self.files = [self.files[i] for i in self.inds_keep]

        if self.labels is not None and len(self.labels) > 0:
            self.labels = self.labels[self.inds_keep]
        self.dataset = torch.utils.data.Subset(
            self.dataset,
            self.inds_keep,
        )

    def _stratified_subsample(self, rng):
        unique_labels, class_counts = np.unique(self.labels, return_counts=True)

        # Fast path: perfectly balanced
        if len(np.unique(class_counts)) == 1:
            logger.info(f"Dataset is balanced ({class_counts[0]} per class)")
            is_balanced = True
        else:
            cv = class_counts.std() / class_counts.mean() * 100
            is_balanced = cv < 5.0
            logger.info(f"Dataset CV: {cv:.2f}% ")
            logger.info(f"Dataset is {'balanced' if is_balanced else 'unbalanced'}")

        n_per_class = self.nsample // len(unique_labels)
        inds_keep = []

        if is_balanced:
            for label in unique_labels:
                class_inds = np.where(self.labels == label)[0]
                selected = rng.choice(
                    class_inds, n_per_class, replace=self.sample_w_replacement
                )
                inds_keep.extend(selected)
        else:
            raw_counts = self.nsample * class_counts / class_counts.sum()
            floor_counts = np.floor(raw_counts).astype(int)
            deficit = self.nsample - floor_counts.sum()
            top_classes = np.argsort(raw_counts - floor_counts)[::-1][:deficit]
            floor_counts[top_classes] += 1

            for label, n in zip(unique_labels, floor_counts):
                class_inds = np.where(self.labels == label)[0]
                n = min(n, len(class_inds)) if not self.sample_w_replacement else n
                selected = rng.choice(class_inds, n, replace=self.sample_w_replacement)
                inds_keep.extend(selected)

        return sorted(inds_keep)

    def get_dataloader(self):
        """
        Create a single overall torch DataLoader over all items and assign it to
        self.dataloader.

        Per-label grouping is handled downstream by filtering the overall
        representations with self.labels, so no per-label loaders are built here.
        """
        self.nimages = len(self.dataset)
        if self.batch_size > self.nimages:
            logger.warning(
                (
                    "Batch size is bigger than the data size. "
                    "Setting batch size to data size"
                )
            )
            self.batch_size = self.nimages

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        if self.labels is not None and len(self.labels) > 0:
            self.label_values = np.unique(self.labels).tolist()
        else:
            self.label_values = None

    def __str__(self):
        if self.labels is not None:
            _, counts = np.unique(self.labels, return_counts=True)
            counts = counts.tolist()
        else:
            counts = "N/A"

        return (
            f"DataModule for path {self.path}\n"
            f"\tdataset name {self.dataset_name}\n"
            f"\timages {self.original_ds_len}, used {self.ds_len}\n"
            f"\tbatch size {self.batch_size}\n"
            f"\timages in loader: {len(self.dataloader.dataset)}"
            f"\tlabels {self.label_values}\n"
            f"\tsamples per label {counts}\n"
        )


def get_datamodule(
    path,
    nsample=-1,
    batch_size=32,
    num_workers=1,
    transform=None,
    seed=0,
    random_sample=True,
    sample_w_replacement=False,
):
    """Deal with format of input path, and get relevant DataLoader"""

    train_str = "test"
    if "--" in path:
        # Path is instead torchvision.dataset
        # e.g. CIFAR10--train, MNIST--test, etc.
        path, train_str = path.split("--")

    train_set = True if train_str.upper() == "TRAIN" else False

    DM = DataModule(
        path,
        train_set=train_set,
        nsample=nsample,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        seed=seed,
        random_sample=random_sample,
        sample_w_replacement=sample_w_replacement,
    )

    return DM


def get_datamodule_from_path(
    path,
    model_transform,
    num_workers,
    args,
    sample_w_replacement=False,
):
    print(f"\nGetting DataModule for path: {path}", file=sys.stderr)

    DM = get_datamodule(
        path,
        args.nsample,
        args.batch_size,
        num_workers,
        seed=args.seed,
        sample_w_replacement=sample_w_replacement,
        transform=lambda x: model_transform(x),
    )

    print(DM)
    return DM
