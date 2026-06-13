"""
Autonomously guide the kiwi robot to face an ArUco marker at 30 cm.

Behavior:
    - Detects marker IDs 0-3.
    - If no target is visible, rotates at 30% speed to search.
    - If visible, uses guided holonomic translation:
        Approach the current marker to 50 cm.
        Slide right while maintaining range until the next marker appears.
    - Commands are acceleration-limited and capped at 50% speed.

Serial protocol:
    <Vx,Vy,Om>\n
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import serial


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_DIR = BASE_DIR / "live_charuco_calibration_output"
TARGET_IDS = {0, 1, 2, 3}

# The 20 cm printable PDF includes a one-module white margin around a 7-module
# ArUco bitmap, so the actual detected black marker is 20cm * 7/9.
DEFAULT_MARKER_LENGTH_M = 0.20 * 7.0 / 9.0

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
}


@dataclass
class MarkerPose:
    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray
    yaw_error: float
    area: float
    image_x: float


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    om: float = 0.0


@dataclass
class FilteredTarget:
    marker_id: int | None = None
    x: float = 0.0
    z: float = 0.0
    yaw_error: float = 0.0
    image_x: float = 0.0
    valid: bool = False
    lost_frames: int = 0
    erratic_frames: int = 0


@dataclass
class SequenceState:
    target_ids: list[int]
    active_index: int = 0
    completed: bool = False
    lateral_searching: bool = False
    guide_from_id: int | None = None

    @property
    def current_target_id(self) -> int | None:
        if self.completed or not self.target_ids:
            return None
        return self.target_ids[self.active_index]

    @property
    def previous_target_id(self) -> int | None:
        if self.guide_from_id is not None:
            return self.guide_from_id
        if self.active_index <= 0:
            return None
        return self.target_ids[self.active_index - 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous ArUco docking controller for kiwi drive.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Requested camera FPS.")
    parser.add_argument("--detect-every", type=int, default=1, help="Run ArUco detection every N frames.")
    parser.add_argument("--no-capture-thread", action="store_true", help="Read frames in the main loop instead of using a latest-frame capture thread.")
    parser.add_argument("--no-roi-tracking", action="store_true", help="Disable ROI-first ArUco detection.")
    parser.add_argument("--roi-padding", type=int, default=80, help="Pixels to pad around the last target marker ROI.")
    parser.add_argument("--roi-min-size", type=int, default=160, help="Minimum ROI width/height in pixels.")
    parser.add_argument("--roi-lost-frames", type=int, default=8, help="Keep trying the ROI for this many missed detection cycles.")
    parser.add_argument(
        "--detect-scale",
        type=float,
        default=1.0,
        help="Scale frame before detection. Try 0.75 or 0.5 for more FPS.",
    )
    parser.add_argument("--calibration", default="", help="Calibration .json/.npz path. Defaults to newest saved file.")
    parser.add_argument("--dictionary", default="", help="ArUco dictionary override.")
    parser.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH_M, help="Black marker side length in meters.")
    parser.add_argument(
        "--target-sequence",
        default="0,2,1,3",
        help="Comma-separated marker IDs to visit in order.",
    )
    parser.add_argument("--serial-port", default=os.environ.get("KIWI_SERIAL_PORT", "/dev/ttyAMA0"))
    parser.add_argument("--baud", type=int, default=int(os.environ.get("KIWI_BAUD_RATE", "115200")))
    parser.add_argument("--target-distance", type=float, default=0.50, help="Desired marker distance on camera Z axis, meters.")
    parser.add_argument("--guide-right-speed", type=float, default=0.50, help="Rightward translation command while searching for the next marker.")
    parser.add_argument("--guide-range-gain", type=float, default=0.65, help="Distance correction gain during guided right translation.")
    parser.add_argument("--guide-x-gain", type=float, default=0.18, help="Weak image/lateral correction while approaching a visible marker.")
    parser.add_argument("--guide-start-yaw-tolerance-deg", type=float, default=20.0, help="Marker 0 must be within this yaw error before starting guided right translation.")
    parser.add_argument("--max-speed", type=float, default=0.50, help="Absolute safety cap for any Vx/Vy/Om command.")
    parser.add_argument("--max-translation-speed", type=float, default=0.50, help="Maximum combined Vx/Vy translation command.")
    parser.add_argument("--min-translation-speed", type=float, default=0.50, help="Minimum combined Vx/Vy translation command while actively correcting.")
    parser.add_argument("--max-rotation-speed", type=float, default=0.14, help="Maximum Om command while tracking.")
    parser.add_argument("--cruise-distance", type=float, default=1.0, help="Above this marker Z distance, prioritize translation over yaw alignment.")
    parser.add_argument("--final-distance", type=float, default=0.55, help="Switch to precise correction inside this marker Z distance.")
    parser.add_argument("--final-translation-speed", type=float, default=0.50, help="Maximum translation command in final correction mode.")
    parser.add_argument("--final-rotation-speed", type=float, default=0.07, help="Maximum rotation command in final correction mode.")
    parser.add_argument("--search-speed", type=float, default=-0.30, help="Rotation speed when no marker is visible.")
    parser.add_argument("--command-hz", type=float, default=20.0, help="Serial command update rate.")
    parser.add_argument("--accel-limit", type=float, default=0.65, help="Max command change per second.")
    parser.add_argument("--x-gain", type=float, default=0.65, help="Lateral centering gain.")
    parser.add_argument("--z-gain", type=float, default=0.38, help="Distance control gain.")
    parser.add_argument("--yaw-gain", type=float, default=0.14, help="Perpendicular alignment gain.")
    parser.add_argument("--cruise-x-gain", type=float, default=0.55, help="Lateral gain while farther than cruise distance.")
    parser.add_argument("--cruise-yaw-gain", type=float, default=0.01, help="Very weak yaw gain while farther than cruise distance.")
    parser.add_argument("--rough-x-gain", type=float, default=0.28, help="Loose lateral gain before final correction.")
    parser.add_argument("--rough-yaw-gain", type=float, default=0.06, help="Loose yaw gain before final correction.")
    parser.add_argument("--slow-distance", type=float, default=0.45, help="Start decelerating translation inside this error distance.")
    parser.add_argument("--pose-filter", type=float, default=0.35, help="Pose smoothing factor from 0-1. Higher reacts faster.")
    parser.add_argument("--pose-lost-grace", type=int, default=5, help="Keep using the last stable pose for this many missed detections.")
    parser.add_argument("--pose-reject-frames", type=int, default=2, help="Reject this many consecutive erratic pose jumps before accepting a new pose.")
    parser.add_argument("--pose-max-jump-x", type=float, default=0.22, help="Reject one-frame lateral pose jumps above this many meters.")
    parser.add_argument("--pose-max-jump-z", type=float, default=0.35, help="Reject one-frame range pose jumps above this many meters.")
    parser.add_argument("--pose-max-jump-image-x", type=float, default=0.48, help="Reject one-frame image-space X jumps above this normalized value.")
    parser.add_argument("--pose-max-jump-yaw-deg", type=float, default=90.0, help="Reject one-frame yaw jumps above this many degrees.")
    parser.add_argument("--alignment-yaw-gain", type=float, default=0.08, help="Continuous weak yaw correction while approaching or guiding.")
    parser.add_argument("--alignment-image-gain", type=float, default=0.16, help="Continuous image-space strafe correction to keep the marker in view.")
    parser.add_argument("--keep-in-view-start", type=float, default=0.62, help="Start prioritizing camera visibility when normalized image X exceeds this magnitude.")
    parser.add_argument("--keep-in-view-hard", type=float, default=0.86, help="Strong keep-in-view recovery when normalized image X exceeds this magnitude.")
    parser.add_argument("--visibility-x-gain", type=float, default=0.42, help="Extra strafe gain from image-space offset while recovering visibility.")
    parser.add_argument("--visibility-min-approach-scale", type=float, default=0.35, help="Minimum forward approach scale while keeping a marker in view.")
    parser.add_argument("--visibility-yaw-scale", type=float, default=0.18, help="Yaw scale at maximum keep-in-view recovery pressure.")
    parser.add_argument("--x-tolerance", type=float, default=0.025, help="Final lateral tolerance in meters.")
    parser.add_argument("--z-tolerance", type=float, default=0.10, help="Final distance tolerance in meters.")
    parser.add_argument("--yaw-tolerance-deg", type=float, default=8.0, help="Final yaw tolerance in degrees.")
    parser.add_argument("--cruise-x-tolerance", type=float, default=0.06, help="Lateral tolerance while farther than cruise distance.")
    parser.add_argument("--cruise-yaw-tolerance-deg", type=float, default=60.0, help="Yaw tolerance while farther than cruise distance.")
    parser.add_argument("--rough-x-tolerance", type=float, default=0.12, help="Loose lateral tolerance before final correction.")
    parser.add_argument("--rough-yaw-tolerance-deg", type=float, default=30.0, help="Loose yaw tolerance before final correction.")
    parser.add_argument("--vx-sign", type=float, default=float(os.environ.get("KIWI_VX_SIGN", "-1")))
    parser.add_argument("--vy-sign", type=float, default=float(os.environ.get("KIWI_VY_SIGN", "1")))
    parser.add_argument("--om-sign", type=float, default=float(os.environ.get("KIWI_OM_SIGN", "1")))
    parser.add_argument(
        "--yaw-control-sign",
        type=float,
        default=1.0,
        help="Invert this if perpendicular yaw correction rotates the wrong way.",
    )
    parser.add_argument("--no-window", action="store_true", help="Do not show the OpenCV preview.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without opening serial.")
    parser.add_argument("--print-every", type=float, default=0.5, help="Seconds between console status prints.")
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def apply_deadband(value: float, deadband: float) -> float:
    return 0.0 if abs(value) < deadband else value


def parse_target_sequence(raw: str) -> list[int]:
    target_ids = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        marker_id = int(text)
        if marker_id not in target_ids:
            target_ids.append(marker_id)
    if not target_ids:
        raise ValueError("Target sequence must include at least one marker ID")
    return target_ids


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

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(data["distortion_coefficients"], dtype=np.float64)
        dictionary_name = data.get("dictionary", "DICT_5X5_1000")
        image_size = (int(data["image_width"]), int(data["image_height"]))
    elif path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
        dictionary_name = "DICT_5X5_1000"
        image_size = tuple(int(v) for v in data["image_size"]) if "image_size" in data else None
    else:
        raise ValueError("Calibration must be .json or .npz")

    print(f"Loaded calibration: {path}")
    return camera_matrix, dist_coeffs, dictionary_name, image_size


def scaled_camera_matrix(camera_matrix: np.ndarray, calibration_size: tuple[int, int] | None, frame_size: tuple[int, int]) -> np.ndarray:
    if calibration_size is None or calibration_size == frame_size:
        return camera_matrix

    calibration_width, calibration_height = calibration_size
    frame_width, frame_height = frame_size
    scaled = camera_matrix.copy()
    scaled[0, 0] *= frame_width / calibration_width
    scaled[0, 2] *= frame_width / calibration_width
    scaled[1, 1] *= frame_height / calibration_height
    scaled[1, 2] *= frame_height / calibration_height
    return scaled


def create_detector(dictionary_name: str):
    if dictionary_name not in DICTIONARIES:
        raise ValueError(f"Unsupported dictionary: {dictionary_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[dictionary_name])
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    return dictionary, parameters


def detect_markers(detector, gray):
    if hasattr(cv2.aruco, "ArucoDetector") and isinstance(detector, cv2.aruco.ArucoDetector):
        return detector.detectMarkers(gray)

    dictionary, parameters = detector
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def detect_markers_scaled(detector, gray: np.ndarray, scale: float):
    if scale >= 0.999:
        return detect_markers(detector, gray)

    scaled_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    corners, ids, rejected = detect_markers(detector, scaled_gray)
    if corners:
        corners = [corner / scale for corner in corners]
    return corners, ids, rejected


def roi_from_corners(
    corners: np.ndarray,
    frame_size: tuple[int, int],
    padding: int,
    min_size: int,
) -> tuple[int, int, int, int]:
    frame_width, frame_height = frame_size
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    min_x = int(math.floor(float(np.min(points[:, 0])))) - padding
    max_x = int(math.ceil(float(np.max(points[:, 0])))) + padding
    min_y = int(math.floor(float(np.min(points[:, 1])))) - padding
    max_y = int(math.ceil(float(np.max(points[:, 1])))) + padding

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width = max(max_x - min_x, min_size)
    height = max(max_y - min_y, min_size)

    x0 = clamp(center_x - width // 2, 0, max(frame_width - 1, 0))
    y0 = clamp(center_y - height // 2, 0, max(frame_height - 1, 0))
    x1 = clamp(x0 + width, 1, frame_width)
    y1 = clamp(y0 + height, 1, frame_height)
    x0 = int(max(0, x1 - width)) if x1 == frame_width else int(x0)
    y0 = int(max(0, y1 - height)) if y1 == frame_height else int(y0)
    return int(x0), int(y0), int(x1), int(y1)


def detect_markers_roi(detector, gray: np.ndarray, scale: float, roi: tuple[int, int, int, int] | None):
    if roi is None:
        corners, ids, rejected = detect_markers_scaled(detector, gray, scale)
        return corners, ids, rejected, False

    x0, y0, x1, y1 = roi
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        corners, ids, rejected = detect_markers_scaled(detector, gray, scale)
        return corners, ids, rejected, False

    corners, ids, rejected = detect_markers_scaled(detector, crop, scale)
    if corners:
        offset = np.array([[[x0, y0]]], dtype=np.float32)
        corners = [corner + offset for corner in corners]
    return corners, ids, rejected, True


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


def estimate_pose(
    corners,
    marker_id: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
    frame_size: tuple[int, int],
) -> MarkerPose | None:
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        marker_object_points(marker_length),
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    marker_normal = rotation_matrix[:, 2].astype(np.float64)

    # For docking we only care that the marker plane is perpendicular to the
    # camera, not which side of the printed marker defines +Z. Square PnP can
    # flip between equivalent front/back-facing normals, which otherwise makes
    # yaw jump by about 180 degrees near the goal and traps the controller in
    # FINAL mode.
    if marker_normal[2] < 0.0:
        marker_normal *= -1.0

    yaw_error = math.atan2(float(marker_normal[0]), float(marker_normal[2]))
    area = float(cv2.contourArea(image_points))
    frame_width, _frame_height = frame_size
    marker_center_x = float(np.mean(image_points[:, 0]))
    image_x = 0.0
    if frame_width > 1:
        image_x = ((marker_center_x / float(frame_width - 1)) * 2.0) - 1.0
    return MarkerPose(marker_id, rvec, tvec, image_points, yaw_error, area, image_x)


def choose_target(poses: list[MarkerPose], desired_id: int | None) -> MarkerPose | None:
    if not poses or desired_id is None:
        return None

    matching = [pose for pose in poses if pose.marker_id == desired_id]
    if not matching:
        return None

    # Prefer the marker nearest the center line, then the closest/largest one.
    return min(
        matching,
        key=lambda pose: (
            abs(float(pose.tvec[0])),
            float(pose.tvec[2]),
            -pose.area,
        ),
    )


def estimate_target_poses(
    corners,
    ids,
    desired_ids: set[int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
    frame_size: tuple[int, int],
) -> list[MarkerPose]:
    if ids is None or not desired_ids:
        return []

    poses = []
    for marker_corners, marker_id_array in zip(corners, ids):
        marker_id = int(marker_id_array[0])
        if marker_id not in desired_ids:
            continue
        pose = estimate_pose(
            marker_corners,
            marker_id,
            camera_matrix,
            dist_coeffs,
            marker_length,
            frame_size,
        )
        if pose is not None:
            poses.append(pose)
    return poses


def angle_delta(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def pose_is_erratic(filtered: FilteredTarget, pose: MarkerPose, args: argparse.Namespace) -> bool:
    if not filtered.valid or filtered.marker_id != pose.marker_id:
        return False

    x, _y, z = pose.tvec
    return (
        abs(float(x) - filtered.x) > args.pose_max_jump_x
        or abs(float(z) - filtered.z) > args.pose_max_jump_z
        or abs(float(pose.image_x) - filtered.image_x) > args.pose_max_jump_image_x
        or abs(math.degrees(angle_delta(float(pose.yaw_error), filtered.yaw_error))) > args.pose_max_jump_yaw_deg
    )


def update_filtered_target(
    filtered: FilteredTarget,
    pose: MarkerPose | None,
    alpha: float,
    args: argparse.Namespace,
) -> FilteredTarget:
    if pose is None:
        filtered.lost_frames += 1
        if filtered.lost_frames > args.pose_lost_grace:
            filtered.valid = False
            filtered.marker_id = None
        return filtered

    x, _y, z = pose.tvec
    if pose_is_erratic(filtered, pose, args):
        filtered.erratic_frames += 1
        filtered.lost_frames = min(filtered.lost_frames + 1, args.pose_lost_grace)
        if filtered.erratic_frames <= args.pose_reject_frames:
            return filtered

    if not filtered.valid or filtered.marker_id != pose.marker_id:
        filtered.marker_id = pose.marker_id
        filtered.x = float(x)
        filtered.z = float(z)
        filtered.yaw_error = float(pose.yaw_error)
        filtered.image_x = float(pose.image_x)
        filtered.valid = True
        filtered.lost_frames = 0
        filtered.erratic_frames = 0
        return filtered

    alpha = clamp(alpha, 0.0, 1.0)
    filtered.x += alpha * (float(x) - filtered.x)
    filtered.z += alpha * (float(z) - filtered.z)
    filtered.yaw_error += alpha * angle_delta(float(pose.yaw_error), filtered.yaw_error)
    filtered.image_x += alpha * (float(pose.image_x) - filtered.image_x)
    filtered.lost_frames = 0
    filtered.erratic_frames = 0
    return filtered


def limit_translation(vx: float, vy: float, limit: float) -> tuple[float, float]:
    magnitude = math.hypot(vx, vy)
    if magnitude <= limit or magnitude <= 1e-6:
        return vx, vy
    scale = limit / magnitude
    return vx * scale, vy * scale


def floor_translation(vx: float, vy: float, floor: float) -> tuple[float, float]:
    magnitude = math.hypot(vx, vy)
    if magnitude <= 1e-6 or magnitude >= floor:
        return vx, vy
    scale = floor / magnitude
    return vx * scale, vy * scale


def keep_in_view_pressure(image_x: float, start: float, hard: float) -> float:
    start = max(0.0, min(start, 0.99))
    hard = max(start + 1e-3, min(hard, 0.999))
    return clamp((abs(image_x) - start) / (hard - start), 0.0, 1.0)


def is_at_target_distance(target: FilteredTarget, args: argparse.Namespace) -> bool:
    return target.valid and target.lost_frames == 0 and abs(target.z - args.target_distance) <= args.z_tolerance


def yaw_is_aligned(target: FilteredTarget, tolerance_deg: float) -> bool:
    return abs(math.degrees(target.yaw_error)) <= tolerance_deg


def apply_translation_limits(vx: float, vy: float, args: argparse.Namespace) -> tuple[float, float]:
    vx, vy = limit_translation(vx, vy, min(args.max_translation_speed, args.max_speed))
    vx, vy = floor_translation(vx, vy, args.min_translation_speed)
    return vx, vy


def command_from_target(
    target: FilteredTarget,
    args: argparse.Namespace,
    require_yaw_lock: bool = False,
) -> tuple[Command, str]:
    if is_at_target_distance(target, args) and (
        not require_yaw_lock or yaw_is_aligned(target, args.guide_start_yaw_tolerance_deg)
    ):
        return Command(0.0, 0.0, 0.0), "LOCKED"

    if not target.valid:
        return Command(0.0, 0.0, args.search_speed), "SEARCH"

    distance_error = target.z - args.target_distance
    vx = args.guide_x_gain * target.x + args.alignment_image_gain * target.image_x
    vy = args.z_gain * distance_error
    om = args.yaw_control_sign * args.alignment_yaw_gain * target.yaw_error
    mode = "APPROACH"

    if require_yaw_lock and is_at_target_distance(target, args):
        vx = 0.0
        vy = 0.0
        mode = "ALIGN"

    visibility_pressure = keep_in_view_pressure(target.image_x, args.keep_in_view_start, args.keep_in_view_hard)
    if visibility_pressure > 0.0:
        vx += args.visibility_x_gain * target.image_x * visibility_pressure
        approach_scale = 1.0 - visibility_pressure * (1.0 - args.visibility_min_approach_scale)
        vy *= approach_scale
        om *= 1.0 - visibility_pressure * (1.0 - args.visibility_yaw_scale)

    if target.lost_frames > 0:
        recovery_scale = clamp(1.0 - target.lost_frames / max(args.pose_lost_grace + 1, 1), 0.25, 1.0)
        vx *= recovery_scale
        vy *= recovery_scale
        om *= recovery_scale
        mode = "RECOVER"

    vx, vy = apply_translation_limits(vx, vy, args)
    command = Command(
        clamp(vx, -args.max_speed, args.max_speed),
        clamp(vy, -args.max_speed, args.max_speed),
        clamp(om, -args.max_rotation_speed, args.max_rotation_speed),
    )
    return command, mode


def guide_right_command(range_target: FilteredTarget, args: argparse.Namespace) -> tuple[Command, str]:
    vx = max(args.guide_right_speed, args.min_translation_speed)
    vy = 0.0
    om = 0.0
    if range_target.valid:
        distance_error = range_target.z - args.target_distance
        vy = 0.0 if abs(distance_error) <= args.z_tolerance else args.guide_range_gain * distance_error
        om = args.yaw_control_sign * args.alignment_yaw_gain * range_target.yaw_error

        visibility_pressure = keep_in_view_pressure(range_target.image_x, args.keep_in_view_start, args.keep_in_view_hard)
        if visibility_pressure > 0.0:
            vx *= 1.0 - visibility_pressure * 0.35
            vy *= 1.0 - visibility_pressure * (1.0 - args.visibility_min_approach_scale)
            om *= 1.0 - visibility_pressure * (1.0 - args.visibility_yaw_scale)

        if range_target.lost_frames > 0:
            recovery_scale = clamp(1.0 - range_target.lost_frames / max(args.pose_lost_grace + 1, 1), 0.35, 1.0)
            vx *= recovery_scale
            vy *= recovery_scale
            om *= recovery_scale

    vx, vy = apply_translation_limits(vx, vy, args)
    return Command(
        clamp(vx, -args.max_speed, args.max_speed),
        clamp(vy, -args.max_speed, args.max_speed),
        clamp(om, -args.max_rotation_speed, args.max_rotation_speed),
    ), "GUIDE-RIGHT" if range_target.lost_frames == 0 else "GUIDE-RECOVER"


def slew_limit(previous: Command, target: Command, max_delta: float) -> Command:
    return Command(
        previous.vx + clamp(target.vx - previous.vx, -max_delta, max_delta),
        previous.vy + clamp(target.vy - previous.vy, -max_delta, max_delta),
        previous.om + clamp(target.om - previous.om, -max_delta, max_delta),
    )


def serial_command(intent: Command, args: argparse.Namespace) -> Command:
    return Command(
        clamp(intent.vx * args.vx_sign, -args.max_speed, args.max_speed),
        clamp(intent.vy * args.vy_sign, -args.max_speed, args.max_speed),
        clamp(intent.om * args.om_sign, -args.max_speed, args.max_speed),
    )


def encode_command(command: Command) -> bytes:
    return f"<{command.vx:.3f},{command.vy:.3f},{command.om:.3f}>\n".encode("ascii")


class RobotSerial:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.serial = None
        if not args.dry_run:
            self.serial = serial.Serial(args.serial_port, args.baud, timeout=0.1, write_timeout=1)
            print(f"Opened robot serial {args.serial_port} @ {args.baud}")

    def write(self, command: Command) -> None:
        frame = encode_command(command)
        if self.serial is not None:
            self.serial.write(frame)

    def close(self) -> None:
        stop = Command()
        for _ in range(5):
            self.write(stop)
            time.sleep(0.03)
        if self.serial is not None:
            self.serial.close()


class LatestFrameCapture:
    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.timestamp = 0.0
        self.error = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None, int, float]:
        deadline = time.monotonic() + timeout
        while not self._stop.is_set():
            with self._lock:
                if self.frame is not None:
                    return True, self.frame.copy(), self.frame_id, self.timestamp
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        return False, None, 0, 0.0

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.error = "Failed to read camera frame"
                time.sleep(0.01)
                continue

            with self._lock:
                self.frame = frame
                self.frame_id += 1
                self.timestamp = time.monotonic()
                self.error = ""


def draw_overlay(
    frame,
    pose: MarkerPose | None,
    intent: Command,
    serial_out: Command,
    mode: str,
    camera_matrix,
    dist_coeffs,
    args,
    sequence: SequenceState,
) -> None:
    sequence_text = " -> ".join(
        f"[{marker_id}]" if marker_id == sequence.current_target_id and not sequence.completed else str(marker_id)
        for marker_id in sequence.target_ids
    )
    if sequence.completed:
        sequence_text = f"{sequence_text} | complete"
    elif sequence.lateral_searching:
        sequence_text = f"{sequence_text} | guide right"

    lines = [
        f"Mode: {mode}",
        f"Sequence: {sequence_text}",
        f"Intent Vx={intent.vx:+.2f} Vy={intent.vy:+.2f} Om={intent.om:+.2f}",
        f"Serial <{serial_out.vx:+.2f},{serial_out.vy:+.2f},{serial_out.om:+.2f}>",
        "Q/ESC quit",
    ]

    if pose is not None:
        x, y, z = pose.tvec
        lines.insert(
            1,
            f"ID {pose.marker_id}: x={x:+.2f} y={y:+.2f} z={z:+.2f} yaw={math.degrees(pose.yaw_error):+.1f} deg",
        )
        lines.insert(2, f"Image X={pose.image_x:+.2f} | keep-in-view {args.keep_in_view_start:.2f}->{args.keep_in_view_hard:.2f}")
        lines.insert(
            3,
            f"Goal: z={args.target_distance:.2f} m +/- {args.z_tolerance:.2f} m | start yaw <= {args.guide_start_yaw_tolerance_deg:.0f} deg",
        )
        cv2.drawFrameAxes(
            frame,
            camera_matrix,
            dist_coeffs,
            pose.rvec.reshape(3, 1),
            pose.tvec.reshape(3, 1),
            args.marker_length * 0.5,
        )

    y_cursor = 28
    for line in lines:
        cv2.putText(frame, line, (12, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y_cursor += 28


def main() -> None:
    args = parse_args()
    sequence = SequenceState(parse_target_sequence(args.target_sequence))
    camera_matrix, dist_coeffs, calibration_dictionary, calibration_size = load_calibration(args.calibration)
    dictionary_name = args.dictionary or calibration_dictionary
    detector = create_detector(dictionary_name)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture = None
    if not args.no_capture_thread:
        capture = LatestFrameCapture(cap)
        capture.start()

    robot = RobotSerial(args)
    command_period = 1.0 / args.command_hz
    previous_intent = Command()
    last_command_time = time.monotonic()
    last_print_time = 0.0
    frame_count = 0
    fps_time = time.monotonic()
    measured_fps = 0.0
    active_camera_matrix = None
    last_poses: list[MarkerPose] = []
    filtered_target = FilteredTarget()
    range_target = FilteredTarget()
    roi_box: tuple[int, int, int, int] | None = None
    roi_misses = args.roi_lost_frames + 1
    last_frame_id = 0

    print("Autonomous ArUco docking controller")
    print(f"Target sequence: {sequence.target_ids}")
    print(f"Target distance: {args.target_distance:.2f} m")
    print(
        "Guided path: approach ID 0, align roughly perpendicular, "
        "then translate right until the next ID appears"
    )
    print(
        f"Max speed: {args.max_speed:.2f}, translation cap: {args.max_translation_speed:.2f}, "
        f"rotation cap: {args.max_rotation_speed:.2f}, final cap: "
        f"{args.final_translation_speed:.2f}/{args.final_rotation_speed:.2f}, search speed: {args.search_speed:.2f}"
    )
    print(f"Phase distances: cruise>{args.cruise_distance:.2f} m, final<{args.final_distance:.2f} m")
    print(f"Requested camera: {args.width}x{args.height} @ {args.camera_fps:g} fps")
    print(
        "Vision optimizations: "
        f"capture_thread={'off' if args.no_capture_thread else 'on'}, "
        f"roi_tracking={'off' if args.no_roi_tracking else 'on'}"
    )
    print("If yaw correction rotates the wrong way, rerun with --yaw-control-sign -1")

    try:
        while True:
            if capture is not None:
                ret, frame, source_frame_id, _frame_time = capture.read()
                if source_frame_id == last_frame_id:
                    time.sleep(0.002)
                    continue
                last_frame_id = source_frame_id
            else:
                ret, frame = cap.read()
            if not ret:
                print(capture.error if capture is not None and capture.error else "Failed to read camera frame")
                continue

            assert frame is not None
            frame_count += 1
            if frame_count % 30 == 0:
                fps_now = time.monotonic()
                measured_fps = 30.0 / max(fps_now - fps_time, 1e-6)
                fps_time = fps_now

            frame_size = (frame.shape[1], frame.shape[0])
            if active_camera_matrix is None:
                active_camera_matrix = scaled_camera_matrix(camera_matrix, calibration_size, frame_size)
                print(
                    "Actual camera:",
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "x",
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "@",
                    cap.get(cv2.CAP_PROP_FPS),
                    "fps",
                )

            should_detect = frame_count % max(args.detect_every, 1) == 0
            if should_detect:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                desired_id = sequence.current_target_id
                relevant_ids = {marker_id for marker_id in (desired_id, sequence.previous_target_id) if marker_id is not None}
                use_roi = (
                    not args.no_roi_tracking
                    and not sequence.lateral_searching
                    and roi_box is not None
                    and roi_misses <= args.roi_lost_frames
                )
                active_roi = roi_box if use_roi else None
                corners, ids, _, roi_used = detect_markers_roi(detector, gray, args.detect_scale, active_roi)

                if ids is not None and len(ids) and not args.no_window:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                poses = estimate_target_poses(
                    corners,
                    ids,
                    relevant_ids,
                    active_camera_matrix,
                    dist_coeffs,
                    args.marker_length,
                    frame_size,
                )

                if roi_used and not choose_target(poses, desired_id):
                    corners, ids, _, _ = detect_markers_roi(detector, gray, args.detect_scale, None)
                    if ids is not None and len(ids) and not args.no_window:
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    poses = estimate_target_poses(
                        corners,
                        ids,
                        relevant_ids,
                        active_camera_matrix,
                        dist_coeffs,
                        args.marker_length,
                        frame_size,
                    )

                detected_pose = choose_target(poses, desired_id)
                if detected_pose is not None:
                    roi_box = roi_from_corners(
                        detected_pose.corners,
                        frame_size,
                        args.roi_padding,
                        args.roi_min_size,
                    )
                    roi_misses = 0
                else:
                    roi_misses += 1
                last_poses = poses
            else:
                poses = last_poses

            target_pose = choose_target(poses, sequence.current_target_id)
            if sequence.completed:
                target_intent = Command()
                mode = "COMPLETE"
            elif sequence.lateral_searching:
                if target_pose is not None:
                    sequence.lateral_searching = False
                    sequence.guide_from_id = None
                    filtered_target = update_filtered_target(FilteredTarget(), target_pose, args.pose_filter, args)
                    target_intent, mode = command_from_target(filtered_target, args)
                    roi_box = roi_from_corners(target_pose.corners, frame_size, args.roi_padding, args.roi_min_size)
                    roi_misses = 0
                else:
                    support_pose = choose_target(poses, sequence.previous_target_id)
                    range_target = update_filtered_target(range_target, support_pose, args.pose_filter, args)
                    target_intent, mode = guide_right_command(range_target, args)
            else:
                filtered_target = update_filtered_target(filtered_target, target_pose, args.pose_filter, args)
                require_yaw_lock = sequence.current_target_id == 0 and sequence.active_index == 0
                target_intent, mode = command_from_target(filtered_target, args, require_yaw_lock=require_yaw_lock)

            if not sequence.completed and not sequence.lateral_searching and mode == "LOCKED":
                if sequence.active_index == len(sequence.target_ids) - 1:
                    sequence.completed = True
                    target_intent = Command()
                    mode = "COMPLETE"
                else:
                    sequence.guide_from_id = sequence.current_target_id
                    sequence.active_index += 1
                    sequence.lateral_searching = True
                    range_target = filtered_target
                    filtered_target = FilteredTarget()
                    roi_box = None
                    roi_misses = args.roi_lost_frames + 1
                    target_intent, mode = guide_right_command(range_target, args)

            now = time.monotonic()
            dt = max(now - last_command_time, command_period)
            if now - last_command_time >= command_period:
                max_delta = args.accel_limit * dt
                previous_intent = slew_limit(previous_intent, target_intent, max_delta)
                out = serial_command(previous_intent, args)
                robot.write(out)
                last_command_time = now

                if now - last_print_time >= args.print_every:
                    if target_pose is not None:
                        print(
                            f"{mode} fps={measured_fps:.1f} target={sequence.current_target_id} seen={target_pose.marker_id} "
                            f"x={filtered_target.x:+.3f} z={filtered_target.z:+.3f} "
                            f"yaw={math.degrees(filtered_target.yaw_error):+.1f} imgx={filtered_target.image_x:+.2f} "
                            f"cmd=<{out.vx:+.2f},{out.vy:+.2f},{out.om:+.2f}>"
                        )
                    else:
                        print(
                            f"{mode} fps={measured_fps:.1f} target={sequence.current_target_id} "
                            f"cmd=<{out.vx:+.2f},{out.vy:+.2f},{out.om:+.2f}>"
                        )
                    last_print_time = now

            if not args.no_window:
                draw_overlay(
                    frame,
                    target_pose,
                    previous_intent,
                    serial_command(previous_intent, args),
                    mode,
                    active_camera_matrix,
                    dist_coeffs,
                    args,
                    sequence,
                )
                cv2.putText(
                    frame,
                    f"FPS {measured_fps:.1f}",
                    (frame.shape[1] - 140, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("ArUco Dock Controller", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    finally:
        robot.close()
        if capture is not None:
            capture.stop()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
