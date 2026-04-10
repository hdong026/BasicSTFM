"""Inspect arrays stored in a .npz file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a .npz file.")
    parser.add_argument("path", help="Path to .npz file.")
    args = parser.parse_args()

    path = Path(args.path)
    loaded = np.load(path, allow_pickle=False)
    try:
        for key in loaded.files:
            array = loaded[key]
            print(
                f"{key}: shape={array.shape}, dtype={array.dtype}, "
                f"min={np.nanmin(array):.6g}, max={np.nanmax(array):.6g}"
            )
    finally:
        loaded.close()


if __name__ == "__main__":
    main()
