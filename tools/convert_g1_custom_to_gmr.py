#!/usr/bin/env python3
"""
Convert g1 custom motion CSVs (already in quaternion form) to the GMR pickle format:
{
    'fps': int,
    'root_pos': np.ndarray (num_frames, 3),
    'root_rot': np.ndarray (num_frames, 4)  # quaternion x, y, z, w
    'dof_pos': np.ndarray (num_frames, num_dofs),
    'local_body_pos': None,
    'link_body_list': None,
}
Assumes CSV columns are:
[root_pos_x, root_pos_y, root_pos_z, root_rot_x, root_rot_y, root_rot_z, root_rot_w, dof...]
"""
import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert g1 custom CSVs to GMR pickle format.")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("data/motions/g1_custom"),
        help="Directory containing CSV motion files.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_gmr",
        help="Suffix to append to the base filename for output pickles (e.g., walk1_gmr.pkl). "
        "Use empty string to overwrite.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=120,
        help="FPS to set in the output (default: 120).",
    )
    return parser.parse_args()


def load_csv_rows(csv_path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if not row:
                continue
            try:
                floats = [float(val) for val in row]
            except ValueError as e:
                raise ValueError(f"Failed to parse row {idx} in {csv_path}: {e}") from e
            rows.append(floats)
    return rows


def convert(csv_path: Path, fps: int, suffix: str):
    rows = load_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    num_cols = len(rows[0])
    if num_cols < 8:
        raise ValueError(f"Expected at least 8 columns in {csv_path}, got {num_cols}")

    # Verify consistent column count.
    for idx, row in enumerate(rows):
        if len(row) != num_cols:
            raise ValueError(f"Inconsistent column count in {csv_path} at row {idx}: {len(row)} vs {num_cols}")

    root_pos = np.array([r[0:3] for r in rows], dtype=np.float32)
    root_rot = np.array([r[3:7] for r in rows], dtype=np.float32)
    dof_pos = np.array([r[7:] for r in rows], dtype=np.float32)

    out_dict = {
        "fps": int(fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    out_name = csv_path.stem + suffix + ".pkl"
    out_path = csv_path.with_name(out_name)
    with out_path.open("wb") as f:
        pickle.dump(out_dict, f)

    print(f"Wrote {out_path} (frames={len(rows)}, dofs={dof_pos.shape[1]}, fps={fps})")


def main():
    args = parse_args()
    folder: Path = args.folder
    suffix: str = args.suffix

    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {folder}")

    for csv_path in csv_files:
        convert(csv_path, args.fps, suffix)


if __name__ == "__main__":
    main()
