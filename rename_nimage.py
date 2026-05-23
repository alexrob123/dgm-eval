"""Rename files under ../out-data: replace 'nimage' with 'nimgs' in basenames.

Usage:
    python rename_nimage.py            # dry-run, prints planned renames
    python rename_nimage.py --apply    # actually rename
"""

import argparse
import os
import sys

ROOT = "./out-data"
OLD = "nimage"
NEW = "nimgs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="perform the renames")
    parser.add_argument(
        "--root", default=ROOT, help=f"directory to walk (default: {ROOT})"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"error: {args.root} is not a directory")

    planned = []
    collisions = []
    for dirpath, _, filenames in os.walk(args.root):
        for name in filenames:
            if OLD not in name:
                continue
            new_name = name.replace(OLD, NEW)
            src = os.path.join(dirpath, name)
            dst = os.path.join(dirpath, new_name)
            if os.path.exists(dst):
                collisions.append((src, dst))
            else:
                planned.append((src, dst))

    for src, dst in planned:
        print(f"{'RENAME' if args.apply else 'PLAN  '}: {src}  ->  {dst}")
    for src, dst in collisions:
        print(f"SKIP (target exists): {src}  ->  {dst}", file=sys.stderr)

    print(
        f"\n{len(planned)} to rename, {len(collisions)} skipped"
        + ("" if args.apply else " (dry-run; pass --apply to execute)")
    )

    if args.apply:
        for src, dst in planned:
            os.rename(src, dst)


if __name__ == "__main__":
    main()
