#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path
from typing import List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert g1 custom motion CSVs to pickle files with frame numbers "
            "in the first column."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("data/motions/g1_custom"),
        help="Directory containing CSV motion files.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "FPS to store in the pickle files. If omitted, an FPS is inferred "
            "from the first-column timestamps; falls back to 60 if inference fails."
        ),
    )
    parser.add_argument(
        "--loop-mode",
        choices=["wrap", "clamp"],
        default="wrap",
        help="Loop mode to store alongside the frames.",
    )
    return parser.parse_args()


def infer_fps(rows: List[List[str]]) -> Optional[float]:
    try:
        times = [float(r[0]) for r in rows if r and r[0] != ""]
    except ValueError:
        return None

    if len(times) < 2:
        return None

    diffs = [b - a for a, b in zip(times[:-1], times[1:]) if b > a]
    if not diffs:
        return None

    mean_dt = sum(diffs) / len(diffs)
    if mean_dt <= 0:
        return None

    return 1.0 / mean_dt


def convert_file(csv_path: Path, fps_arg: Optional[float], loop_mode_val: int) -> None:
    with csv_path.open() as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]

    fps = fps_arg or infer_fps(rows) or 60.0

    frames = []
    for idx, row in enumerate(rows):
        try:
            floats = [float(val) for val in row]
        except ValueError as e:
            raise ValueError(f"Failed parsing {csv_path} row {idx}: {e}") from e
        floats[0] = float(idx)  # overwrite first column with frame index
        frames.append(floats)

    out_dict = {
        "loop_mode": loop_mode_val,
        "fps": fps,
        "frames": frames,
    }

    out_path = csv_path.with_suffix(".pkl")
    with out_path.open("wb") as out_f:
        pickle.dump(out_dict, out_f)

    print(f"Wrote {out_path} (frames={len(frames)}, fps={fps:.3f})")


def main():
    args = parse_args()
    folder: Path = args.folder
    loop_mode_val = 1 if args.loop_mode == "wrap" else 0

    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {folder}")

    for csv_path in csv_files:
        convert_file(csv_path, args.fps, loop_mode_val)


if __name__ == "__main__":
    main()
