#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay the first frame of every episode in a LeRobot dataset."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset root. If it does not contain meta/info.json, the script also checks the local LeRobot cache using the folder name.",
    )
    parser.add_argument(
        "--camera-key",
        default="observation.images.front",
        help="Camera key to visualize. Default: observation.images.front",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional PNG output path. Defaults to <dataset>/overlay_<camera>.png",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a matplotlib window; only save the output image.",
    )
    return parser.parse_args()


def resolve_dataset_root(dataset_path: str) -> Path:
    raw_path = Path(dataset_path).expanduser()
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.append((Path.cwd() / raw_path).resolve())

    cache_root = Path.home() / ".cache" / "huggingface" / "lerobot" / "local"
    candidates.append(cache_root / raw_path.name)

    for candidate in candidates:
        if (candidate / "meta" / "info.json").is_file():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find a LeRobot dataset root.\nChecked:\n{checked}")


def load_info(dataset_root: Path) -> dict:
    with open(dataset_root / "meta" / "info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_episode_table(dataset_root: Path) -> pd.DataFrame:
    episodes_dir = dataset_root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {episodes_dir}")

    frames = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    if "episode_index" not in df.columns:
        raise ValueError("Episode metadata is missing the episode_index column.")

    return df.sort_values("episode_index", kind="stable").reset_index(drop=True)


def get_video_path(
    dataset_root: Path,
    info: dict,
    camera_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    video_template = info["video_path"]
    relative_path = video_template.format(
        video_key=camera_key,
        chunk_index=int(chunk_index),
        file_index=int(file_index),
    )
    return dataset_root / relative_path


def read_video_frame(video_path: Path, frame_index: int) -> tuple:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb.shape[:2], rgb
    finally:
        capture.release()


def build_overlay(dataset_root: Path, info: dict, episodes_df: pd.DataFrame, camera_key: str):
    chunk_col = f"videos/{camera_key}/chunk_index"
    file_col = f"videos/{camera_key}/file_index"
    ts_col = f"videos/{camera_key}/from_timestamp"

    missing = [col for col in (chunk_col, file_col, ts_col) if col not in episodes_df.columns]
    if missing:
        raise ValueError(f"Episode metadata is missing columns for {camera_key}: {missing}")

    fps = float(info["fps"])
    total = None
    image_shape = None
    per_episode = []

    for _, row in episodes_df.iterrows():
        video_path = get_video_path(
            dataset_root=dataset_root,
            info=info,
            camera_key=camera_key,
            chunk_index=row[chunk_col],
            file_index=row[file_col],
        )
        from_timestamp = float(row[ts_col])
        frame_index = max(0, int(round(from_timestamp * fps)))
        shape, frame = read_video_frame(video_path, frame_index)

        if image_shape is None:
            image_shape = shape
            total = frame.astype("float32")
        else:
            if shape != image_shape:
                raise ValueError(
                    f"Frame shape mismatch: expected {image_shape}, got {shape} for {video_path}"
                )
            total += frame.astype("float32")

        per_episode.append(
            {
                "episode_index": int(row["episode_index"]),
                "video_path": str(video_path),
                "frame_index": frame_index,
                "from_timestamp": from_timestamp,
            }
        )

    if total is None:
        raise ValueError("No episode frames were loaded.")

    overlay = (total / len(per_episode)).clip(0, 255).astype("uint8")
    return overlay, per_episode


def default_output_path(dataset_root: Path, camera_key: str) -> Path:
    safe_camera = camera_key.replace(".", "_")
    return dataset_root / f"overlay_{safe_camera}.png"


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_path)
    info = load_info(dataset_root)

    if args.camera_key not in info["features"]:
        available = ", ".join(
            key for key, value in info["features"].items() if value.get("dtype") in {"video", "image"}
        )
        raise ValueError(
            f"Camera key '{args.camera_key}' not found in dataset features. Available image/video keys: {available}"
        )

    episodes_df = load_episode_table(dataset_root)
    overlay, per_episode = build_overlay(dataset_root, info, episodes_df, args.camera_key)

    output_path = Path(args.output) if args.output else default_output_path(dataset_root, args.camera_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Dataset root: {dataset_root}")
    print(f"Camera key: {args.camera_key}")
    print(f"Episodes processed: {len(per_episode)}")
    print(f"Saved overlay: {output_path}")

    if not args.no_show:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required for interactive display. Re-run with --no-show or install matplotlib."
            ) from exc

        plt.figure(figsize=(10, 7))
        plt.imshow(overlay)
        plt.title(f"{dataset_root.name}: first-frame overlay ({args.camera_key})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
