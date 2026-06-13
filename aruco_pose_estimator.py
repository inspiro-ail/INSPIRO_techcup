"""
Estimate 3D position of ArUco marker IDs 0-3 using saved ChArUco calibration.

Default usage:
    python aruco_pose_estimator.py

Important:
    Set --marker-length to the physical printed marker side length in meters.
    The default is 0.022 m because char.py used MARKER_LENGTH = 0.022.

Camera coordinate convention:
    X: right in image
    Y: down in image
    Z: forward from camera
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_DIR = BASE_DIR / "live_charuco_calibration_output"
DEFAULT_CAMERA_INDEX = 1
DEFAULT_MARKER_LENGTH_M = 0.022
TARGET_IDS = {0, 1, 2, 3}

DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate 3D pose for ArUco marker IDs 0-3.")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="OpenCV camera index.")
    parser.add_argument(
        "--calibration",
        default="",
        help="Path to calibration .json or .npz. Defaults to newest file in live_charuco_calibration_output.",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=DEFAULT_MARKER_LENGTH_M,
        help="Printed marker side length in meters.",
    )
    parser.add_argument(
        "--dictionary",
        default="",
        help="ArUco dictionary name. Defaults to dictionary stored in calibration, else DICT_5X5_1000.",
    )
    parser.add_argument("--width", type=int, default=0, help="Optional requested camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional requested camera height.")
    parser.add_argument("--axis-length", type=float, default=0.015, help="Drawn pose axis length in meters.")
    parser.add_argument("--print-every", type=float, default=0.2, help="Seconds between console pose prints.")
    parser.add_argument("--no-window", action="store_true", help="Print only; do not show OpenCV preview.")
    parser.add_argument("--plot-3d", action="store_true", help="Show live Matplotlib 3D marker position preview.")
    parser.add_argument("--plot-range", type=float, default=0.6, help="3D preview axis half-range in meters.")
    return parser.parse_args()


def newest_calibration_file() -> Path:
    candidates = list(DEFAULT_CALIBRATION_DIR.glob("calibration_*.npz"))
    candidates += list(DEFAULT_CALIBRATION_DIR.glob("calibration_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No calibration files found in {DEFAULT_CALIBRATION_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_calibration(path_arg: str) -> tuple[np.ndarray, np.ndarray, str, tuple[int, int] | None]:
    path = Path(path_arg).expanduser() if path_arg else newest_calibration_file()
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
        dictionary_name = "DICT_5X5_1000"
        image_size = tuple(int(v) for v in data["image_size"]) if "image_size" in data else None
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(data["distortion_coefficients"], dtype=np.float64)
        dictionary_name = data.get("dictionary", "DICT_5X5_1000")
        image_size = (int(data["image_width"]), int(data["image_height"]))
    else:
        raise ValueError("Calibration must be a .json or .npz file")

    print(f"Loaded calibration: {path}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients shape: {dist_coeffs.shape}")
    return camera_matrix, dist_coeffs, dictionary_name, image_size


def create_detector(dictionary_name: str):
    if dictionary_name not in DICTIONARIES:
        raise ValueError(f"Unsupported dictionary '{dictionary_name}'. Known: {', '.join(DICTIONARIES)}")

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[dictionary_name])
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    return dictionary, parameters


def detect_markers(detector, gray):
    if hasattr(cv2.aruco, "ArucoDetector") and isinstance(detector, cv2.aruco.ArucoDetector):
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids, rejected


def marker_object_points(marker_length: float) -> np.ndarray:
    half = marker_length / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def estimate_marker_pose(corners, camera_matrix, dist_coeffs, marker_length: float):
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    object_points = marker_object_points(marker_length)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not ok:
        return None, None

    return rvec, tvec


def draw_pose_label(frame, marker_id: int, tvec: np.ndarray, corners) -> None:
    x, y, z = tvec.reshape(3)
    corner = np.asarray(corners).reshape(4, 2)[0].astype(int)
    label = f"ID {marker_id}: x={x:+.3f} y={y:+.3f} z={z:+.3f} m"
    cv2.putText(
        frame,
        label,
        (int(corner[0]), max(24, int(corner[1]) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


class MatplotlibPosePreview:
    def __init__(self, plot_range: float) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.plot_range = plot_range
        self.fig = plt.figure("ArUco Marker 3D Positions")
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.ion()
        self.fig.show()

    def update(self, detected_poses: list[tuple[int, np.ndarray]]) -> None:
        self.ax.clear()
        limit = self.plot_range

        self.ax.set_title("Marker positions in camera coordinates")
        self.ax.set_xlabel("X right (m)")
        self.ax.set_ylabel("Z forward (m)")
        self.ax.set_zlabel("Y down (m)")
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(0.0, limit * 2.0)
        self.ax.set_zlim(limit, -limit)

        self.ax.scatter([0.0], [0.0], [0.0], c="black", marker="^", s=80, label="camera")
        self.ax.quiver(0, 0, 0, 0.12, 0, 0, color="red")
        self.ax.quiver(0, 0, 0, 0, 0.12, 0, color="green")
        self.ax.quiver(0, 0, 0, 0, 0, 0.12, color="blue")

        for marker_id, position in sorted(detected_poses):
            x, y, z = position
            self.ax.scatter([x], [z], [y], s=90)
            self.ax.text(x, z, y, f"ID {marker_id}", fontsize=10)

        self.ax.legend(loc="upper right")
        self.plt.pause(0.001)

    def close(self) -> None:
        self.plt.ioff()
        self.plt.close(self.fig)


def main() -> None:
    args = parse_args()
    camera_matrix, dist_coeffs, calibration_dictionary, image_size = load_calibration(args.calibration)
    dictionary_name = args.dictionary or calibration_dictionary
    detector = create_detector(dictionary_name)
    pose_preview = MatplotlibPosePreview(args.plot_range) if args.plot_3d else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"Dictionary: {dictionary_name}")
    print(f"Target IDs: {sorted(TARGET_IDS)}")
    print(f"Marker length: {args.marker_length} m")
    if image_size:
        print(f"Calibration image size: {image_size[0]}x{image_size[1]}")
    print("Camera coordinates: +X right, +Y down, +Z forward. Press Q/ESC to quit.")

    last_print = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(detector, gray)
        now = time.monotonic()
        detected_poses = []

        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for marker_corners, marker_id_array in zip(corners, ids):
                marker_id = int(marker_id_array[0])
                if marker_id not in TARGET_IDS:
                    continue

                rvec, tvec = estimate_marker_pose(
                    marker_corners,
                    camera_matrix,
                    dist_coeffs,
                    args.marker_length,
                )
                if rvec is None or tvec is None:
                    continue

                detected_poses.append((marker_id, tvec.reshape(3)))
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, args.axis_length)
                draw_pose_label(frame, marker_id, tvec, marker_corners)

        if detected_poses and now - last_print >= args.print_every:
            for marker_id, position in sorted(detected_poses):
                x, y, z = position
                print(f"ID {marker_id}: x={x:+.4f} m, y={y:+.4f} m, z={z:+.4f} m")
            last_print = now

        if pose_preview is not None:
            pose_preview.update(detected_poses)

        if not args.no_window:
            cv2.imshow("ArUco 3D Position IDs 0-3", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    if pose_preview is not None:
        pose_preview.close()


if __name__ == "__main__":
    main()
