import os


def make_str(desc):
    out_str = ""

    for k, v in desc.items():
        if k in ["real_ds"]:
            out_str += v
            out_str += "-vs-"
        elif k in ["gen_ds"]:
            out_str += v
            out_str += "_"
        elif k in ["model", "scores"]:
            out_str += v
            out_str += "_"
        else:
            out_str += f"{k}-{v}"
            out_str += "_"

    if out_str.endswith("_"):
        out_str = out_str[:-1]

    return out_str.replace("-_", "_")


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
