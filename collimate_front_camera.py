#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageTk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display live camera 2 with the first observation.images.front frame overlaid for collimation."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset root. If needed, the local LeRobot cache is also checked by folder name.",
    )
    parser.add_argument("--camera-index", type=int, default=2, help="Live camera index. Default: 2")
    parser.add_argument("--camera-key", default="observation.images.front")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Live-view alpha. 0.0 = only dataset frame, 1.0 = only live camera.",
    )
    return parser.parse_args()


def resolve_dataset_root(dataset_path: str) -> Path:
    raw_path = Path(dataset_path).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append((Path.cwd() / raw_path).resolve())
    candidates.append(Path.home() / ".cache" / "huggingface" / "lerobot" / "local" / raw_path.name)

    for candidate in candidates:
        if (candidate / "meta" / "info.json").is_file():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find a LeRobot dataset root.\nChecked:\n{checked}")


def load_info(dataset_root: Path) -> dict:
    return json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))


def load_episode_table(dataset_root: Path) -> pd.DataFrame:
    parquet_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No episode metadata parquet files found.")
    return pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True).sort_values(
        "episode_index", kind="stable"
    )


def read_first_reference_frame(dataset_root: Path, info: dict, episodes_df: pd.DataFrame, camera_key: str):
    first_episode = episodes_df.iloc[0]
    chunk_col = f"videos/{camera_key}/chunk_index"
    file_col = f"videos/{camera_key}/file_index"
    ts_col = f"videos/{camera_key}/from_timestamp"

    missing = [col for col in (chunk_col, file_col, ts_col) if col not in episodes_df.columns]
    if missing:
        raise ValueError(f"Episode metadata is missing columns for {camera_key}: {missing}")

    video_rel = info["video_path"].format(
        video_key=camera_key,
        chunk_index=int(first_episode[chunk_col]),
        file_index=int(first_episode[file_col]),
    )
    video_path = dataset_root / video_rel
    frame_index = max(0, int(round(float(first_episode[ts_col]) * float(info["fps"]))))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open reference video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        return frame
    finally:
        capture.release()


class CollimationApp:
    def __init__(self, args: argparse.Namespace, dataset_root: Path, reference_bgr):
        self.args = args
        self.dataset_root = dataset_root
        self.alpha = max(0.0, min(1.0, args.alpha))
        self.reference_bgr = cv2.resize(reference_bgr, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

        self.cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open live camera index {args.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.cap.set(cv2.CAP_PROP_FPS, args.fps)
        if args.fourcc and len(args.fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc.upper()))

        self.root = tk.Tk()
        self.root.title(f"MiniFab Collimation: {dataset_root.name}")
        self.root.configure(bg="#edf5ff")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("[", lambda _event: self.adjust_alpha(-0.05))
        self.root.bind("]", lambda _event: self.adjust_alpha(0.05))

        self.image_label = tk.Label(self.root, bg="#edf5ff", bd=0)
        self.image_label.pack(padx=12, pady=(12, 8))

        controls = tk.Frame(self.root, bg="#edf5ff")
        controls.pack(fill="x", padx=12, pady=(0, 12))

        self.alpha_label = tk.Label(
            controls,
            text=self._alpha_text(),
            font=("Segoe UI", 11),
            fg="#234f88",
            bg="#edf5ff",
        )
        self.alpha_label.pack(anchor="w", pady=(0, 8))

        self.alpha_scale = tk.Scale(
            controls,
            from_=0,
            to=100,
            orient="horizontal",
            showvalue=False,
            highlightthickness=0,
            bd=0,
            troughcolor="#dceaff",
            activebackground="#7eaef6",
            bg="#edf5ff",
            command=self.on_scale,
        )
        self.alpha_scale.set(int(round(self.alpha * 100)))
        self.alpha_scale.pack(fill="x")

        hint = tk.Label(
            controls,
            text="Use the slider or [ and ] to adjust blend. Esc closes the window.",
            font=("Segoe UI", 10),
            fg="#5a7ea8",
            bg="#edf5ff",
        )
        hint.pack(anchor="w", pady=(8, 0))

        self._photo = None
        self._closed = False

    def _alpha_text(self) -> str:
        return f"Dataset: {self.dataset_root.name}   Live alpha: {self.alpha:.2f}"

    def on_scale(self, value: str) -> None:
        self.alpha = max(0.0, min(1.0, float(value) / 100.0))
        self.alpha_label.configure(text=self._alpha_text())

    def adjust_alpha(self, delta: float) -> None:
        self.alpha = max(0.0, min(1.0, self.alpha + delta))
        self.alpha_scale.set(int(round(self.alpha * 100)))
        self.alpha_label.configure(text=self._alpha_text())

    def update_frame(self) -> None:
        if self._closed:
            return

        ok, live = self.cap.read()
        if ok and live is not None:
            live = cv2.resize(live, (self.args.width, self.args.height), interpolation=cv2.INTER_LINEAR)
            blended = cv2.addWeighted(live, self.alpha, self.reference_bgr, 1.0 - self.alpha, 0.0)
            rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=image)
            self.image_label.configure(image=self._photo)

        self.root.after(max(1, int(1000 / max(1, self.args.fps))), self.update_frame)

    def close(self) -> None:
        self._closed = True
        self.cap.release()
        self.root.destroy()

    def run(self) -> None:
        self.update_frame()
        self.root.mainloop()


def main() -> int:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_path)
    info = load_info(dataset_root)
    if args.camera_key not in info["features"]:
        raise ValueError(f"Camera key '{args.camera_key}' not found in dataset info.")

    episodes_df = load_episode_table(dataset_root)
    reference_bgr = read_first_reference_frame(dataset_root, info, episodes_df, args.camera_key)
    app = CollimationApp(args, dataset_root, reference_bgr)
    app.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
