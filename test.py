import sys
from argparse import Namespace

from dgm_eval.dataloaders import get_dataloader


def get_dataloader_from_path(
    path,
    model_transform,
    num_workers,
    args,
    sample_w_replacement=False,
):
    print(f"Getting DataLoader for path: {path}\n", file=sys.stderr)

    dataloader = get_dataloader(
        path,
        args.nsample,
        args.batch_size,
        num_workers,
        seed=args.seed,
        sample_w_replacement=sample_w_replacement,
        transform=lambda x: model_transform(x),
    )

    return dataloader


args = Namespace(
    path="CIFAR10--train",
    seed=13579,
    random_sample=True,
    nsample=-1,
    batch_size=64,
)


if __name__ == "__main__":
    dataloader_real = get_dataloader_from_path(
        "CIFAR10--train",
        None,
        num_workers=1,
        args=args,
    )

    print(dataloader_real)

    print(dataloader_real.data_set)
    print(len(dataloader_real.data_set))

    print(dataloader_real.data_set.data.shape)
    # print(dataloader_real.data_set.targets)

    print(dataloader_real.data_loader)
    print(len(dataloader_real.data_loader))

    # print(next(iter(dataloader_real.data_loader)))

    print(dataloader_real.labels)
