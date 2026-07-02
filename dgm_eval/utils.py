import os

from dgm_eval.metrics import METRICS


def make_str(desc):
    out_str = ""

    for k, v in desc.items():
        if k in ["train_ds"]:
            out_str += v.replace("-", "_")
            out_str += "_vs_"
        elif k in ["gen_ds"]:
            out_str += v.replace("-", "_")
            out_str += "-"
        elif k in ["model", "metrics"]:
            out_str += v.replace("-", "_")
            out_str += "-"
        elif v is None:
            out_str += f"{k}_None".replace("-", "_")
            out_str += "-"
        elif not v:
            out_str += f"{k}".replace("-", "_")
            out_str += "-"
        else:
            out_str += f"{k}_{v}".replace("-", "_")
            out_str += "-"

    if out_str.endswith("-"):
        out_str = out_str[:-1]

    return out_str.replace("_-", "-")


def get_metric_substring(stem, metric=None):
    if metric is None:
        for m in METRICS:
            try:
                return get_metric_substring(stem, m)
            except ValueError:
                continue
        raise ValueError(f"No metric from METRICS found in '{stem}'")

    for s in stem.split("-"):
        if metric in s:
            return s
    raise ValueError(f"No substring containing '{metric}' in '{stem}'")


def get_nearest_k_substring(stem):
    for s in stem.split("-"):
        if s.startswith("k_"):
            return s
    raise ValueError(f"No substring starting with 'k_' in '{stem}'")


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

    return "_".join(reversed(parts))
