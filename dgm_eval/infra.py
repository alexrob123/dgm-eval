import os
import sys

import torch


def get_device_and_num_workers(device, num_workers):
    if device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(device)

    if num_workers is None:
        try:
            # os. generally works on unix systems
            num_avail_cpus = len(os.sched_getaffinity(0))
        except:
            # Windows or other systems do not support os.sched_getaffinity.
            # And dataloader can have issues with num_workers>0.
            # https://discuss.pytorch.org/t/eoferror-ran-out-of-input-when-enumerating-the-train-loader/22692/7
            print(
                "\nWarning: os.sched_getaffinity error. Limited number of CPUs/workers used.\n",
                file=sys.stderr,
            )
            num_avail_cpus = 0
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = num_workers

    return device, num_workers
