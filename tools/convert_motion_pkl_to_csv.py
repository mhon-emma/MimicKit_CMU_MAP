#!/usr/bin/env python3
"""Convert a MimicKit motion .pkl file (with `frames`) to a CSV."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Make local packages importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mimickit"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a MimicKit motion PKL (dict with 'frames') to CSV."
    )
    parser.add_argument(
        "input_pkl",
        type=Path,
        help="Path to the motion .pkl file (e.g. data/motions/g1/g1_walk.pkl).",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        type=Path,
        help="Optional output CSV path. Defaults to <input_stem>.csv next to the PKL.",
    )
    parser.add_argument(
        "--char-file",
        type=Path,
        default=None,
        help=(
            "Optional MJCF character file to name DOF columns "
            "(e.g. data/assets/g1/g1.xml). If omitted, falls back to joint_i."
        ),
    )
    parser.add_argument(
        "--no-frame-index",
        action="store_true",
        help="Do not include the frame index as the first CSV column.",
    )
    return parser.parse_args()


def load_frames(path: Path) -> Tuple[List[Iterable[float]], int]:
    with path.open("rb") as fh:
        data = pickle.load(fh)

    if not isinstance(data, dict) or "frames" not in data:
        raise ValueError("Expected a dict with a 'frames' key inside the PKL.")

    frames = data["frames"]
    if not frames:
        return [], 0

    joint_count = len(frames[0])
    for idx, frame in enumerate(frames):
        if len(frame) != joint_count:
            raise ValueError(
                f"Frame {idx} has {len(frame)} values, expected {joint_count}."
            )
    return frames, joint_count


def _dof_headers_from_char(char_file: Path, expected_dof: int) -> Optional[Sequence[str]]:
    """Return a sequence of DOF names (length == expected_dof) using the MJCF."""
    try:
        import torch
        from mimickit.anim.kin_char_model import KinCharModel
    except Exception:
        return None

    model = KinCharModel(torch.device("cpu"))
    model.load_char_file(char_file)

    names: list[str] = [""] * expected_dof
    for j in range(1, model.get_num_joints()):
        joint = model.get_joint(j)
        dof_dim = joint.get_dof_dim()
        if dof_dim == 0:
            continue
        base_idx = joint.dof_idx

        if dof_dim == 1:
            names[base_idx] = joint.name
        else:
            # Rare case: spherical joint (3 dof)
            suffixes = ["x", "y", "z"]
            for k in range(dof_dim):
                names[base_idx + k] = f"{joint.name}_{suffixes[k]}"

    # Ensure we filled all slots; otherwise fall back to numbered names.
    if any(not n for n in names):
        return None
    return names


def build_headers(
    joint_count: int, include_index: bool, char_file: Optional[Path]
) -> list[str]:
    headers = []
    if include_index:
        headers.append("frame")

    # First 6 values are root position (3) and root rotation exp-map (3).
    headers.extend(
        [
            "root_pos_x",
            "root_pos_y",
            "root_pos_z",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
        ]
    )

    dof_count = joint_count - 6
    dof_headers: Optional[Sequence[str]] = None
    if char_file is not None and char_file.exists():
        dof_headers = _dof_headers_from_char(char_file, dof_count)
    if dof_headers is None:
        dof_headers = [f"joint_{i}" for i in range(dof_count)]

    headers.extend(dof_headers)
    return headers


def guess_char_file(input_path: Path) -> Optional[Path]:
    """Infer a matching MJCF file from the motion path (e.g., motions/g1 -> assets/g1/g1.xml)."""
    try:
        char_dir = input_path.parent.name
        candidate = REPO_ROOT / "data" / "assets" / char_dir / f"{char_dir}.xml"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    return None


def write_csv(
    frames: List[Iterable[float]],
    joint_count: int,
    out_path: Path,
    include_index: bool,
    char_file: Optional[Path],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = build_headers(joint_count, include_index, char_file)

    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for idx, frame in enumerate(frames):
            row = list(frame)
            if include_index:
                row.insert(0, idx)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_path = args.input_pkl
    output_path = args.output_csv or input_path.with_suffix(".csv")
    char_file = args.char_file or guess_char_file(input_path)

    frames, joint_count = load_frames(input_path)
    write_csv(
        frames,
        joint_count,
        output_path,
        include_index=not args.no_frame_index,
        char_file=char_file,
    )
    extra = f" (headers from {char_file})" if char_file else ""
    print(f"Wrote {len(frames)} frames x {joint_count} joints to {output_path}{extra}")


if __name__ == "__main__":
    main()
