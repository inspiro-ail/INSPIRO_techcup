import cv2
import numpy as np
import json
import os
from datetime import datetime

# ============================================================
# EDIT THESE TO MATCH YOUR PRINTED CHARUCO BOARD
# ============================================================
CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

SQUARES_X = 10
SQUARES_Y = 7

# Use measured printed sizes in meters
SQUARE_LENGTH = 0.030   # example: 30 mm
MARKER_LENGTH = 0.022   # example: 22 mm

DICTIONARY_NAME = "DICT_5X5_1000"
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

# Use "standard" first. Use "fisheye" for very wide / fisheye lenses.
CALIBRATION_MODEL = "standard"
# CALIBRATION_MODEL = "fisheye"

MIN_CHARUCO_CORNERS = 12
MIN_CAPTURES_TO_CALIBRATE = 15

OUTPUT_DIR = "live_charuco_calibration_output"

# ============================================================
# SETUP
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "captures"), exist_ok=True)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    DICTIONARY
)

detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(DICTIONARY, detector_params)

all_charuco_corners = []
all_charuco_ids = []
captured_images = []

camera_matrix = None
dist_coeffs = None
fisheye_K = None
fisheye_D = None
image_size = None

undistort_preview = False
calibrated = False

# ============================================================
# HELPERS
# ============================================================
def draw_text_panel(frame, lines):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    panel_height = 28 * len(lines) + 16
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += 28


def detect_charuco(gray, frame_for_drawing=None):
    marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) < 4:
        return 0, None, None, marker_corners, marker_ids

    cv2.aruco.refineDetectedMarkers(
        image=gray,
        board=board,
        detectedCorners=marker_corners,
        detectedIds=marker_ids,
        rejectedCorners=rejected
    )

    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board
    )

    if frame_for_drawing is not None:
        cv2.aruco.drawDetectedMarkers(frame_for_drawing, marker_corners, marker_ids)

        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                frame_for_drawing,
                charuco_corners,
                charuco_ids,
                (0, 255, 0)
            )

    return num_corners, charuco_corners, charuco_ids, marker_corners, marker_ids


def calibrate_standard(image_size):
    flags = cv2.CALIB_RATIONAL_MODEL

    rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags
    )

    return rms, K, D, rvecs, tvecs


def calibrate_fisheye(image_size):
    obj_points = []
    img_points = []

    board_corners = board.getChessboardCorners()

    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        objp = []
        imgp = []

        for corner, corner_id in zip(corners, ids.flatten()):
            objp.append(board_corners[corner_id])
            imgp.append(corner[0])

        objp = np.array(objp, dtype=np.float32).reshape(-1, 1, 3)
        imgp = np.array(imgp, dtype=np.float32).reshape(-1, 1, 2)

        if len(objp) >= MIN_CHARUCO_CORNERS:
            obj_points.append(objp)
            img_points.append(imgp)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points,
        img_points,
        image_size,
        K,
        D,
        None,
        None,
        flags,
        criteria=(
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6
        )
    )

    return rms, K, D, rvecs, tvecs


def make_standard_undistort_maps(K, D, image_size):
    w, h = image_size

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        D,
        (w, h),
        alpha=0.0
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        K,
        D,
        None,
        new_K,
        (w, h),
        cv2.CV_16SC2
    )

    return map1, map2, new_K


def make_fisheye_undistort_maps(K, D, image_size):
    w, h = image_size

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        (w, h),
        np.eye(3),
        balance=0.3
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        new_K,
        (w, h),
        cv2.CV_16SC2
    )

    return map1, map2, new_K


def save_calibration(model, rms, K, D, image_size):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        "model": model,
        "timestamp": timestamp,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_length_m": SQUARE_LENGTH,
        "marker_length_m": MARKER_LENGTH,
        "dictionary": DICTIONARY_NAME,
        "rms_reprojection_error": float(rms),
        "camera_matrix": K.tolist(),
        "distortion_coefficients": D.tolist(),
        "num_captures": len(all_charuco_corners)
    }

    json_path = os.path.join(OUTPUT_DIR, f"calibration_{model}_{timestamp}.json")
    npz_path = os.path.join(OUTPUT_DIR, f"calibration_{model}_{timestamp}.npz")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    np.savez(
        npz_path,
        model=model,
        image_size=image_size,
        camera_matrix=K,
        dist_coeffs=D,
        rms=rms
    )

    print(f"Saved {json_path}")
    print(f"Saved {npz_path}")


# ============================================================
# MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

# Request the same resolution you plan to use for marker tracking.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

map1 = None
map2 = None
new_camera_matrix = None
latest_rms = None

print("Controls:")
print("SPACE = capture current frame")
print("C     = calibrate")
print("U     = toggle undistorted preview")
print("S     = save calibration")
print("R     = reset captures")
print("Q/ESC = quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read from camera")
        break

    h, w = frame.shape[:2]
    image_size = (w, h)

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_corners, charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
        gray,
        display
    )

    detection_good = (
        charuco_corners is not None
        and charuco_ids is not None
        and num_corners >= MIN_CHARUCO_CORNERS
    )

    if undistort_preview and calibrated and map1 is not None and map2 is not None:
        display = cv2.remap(display, map1, map2, cv2.INTER_LINEAR)

    status = "GOOD" if detection_good else "MOVE BOARD / NEED MORE CORNERS"

    lines = [
        f"ChArUco corners: {num_corners} | Status: {status}",
        f"Captured frames: {len(all_charuco_corners)} | Need at least: {MIN_CAPTURES_TO_CALIBRATE}",
        f"Model: {CALIBRATION_MODEL} | Calibrated: {calibrated} | RMS: {latest_rms}",
        "SPACE capture | C calibrate | U undistort preview | S save | R reset | Q quit"
    ]

    draw_text_panel(display, lines)

    cv2.imshow("Live ChArUco Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    elif key == ord(" "):
        if detection_good:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_path = os.path.join(
                OUTPUT_DIR,
                "captures",
                f"capture_{len(all_charuco_corners):03d}_{timestamp}.png"
            )
            cv2.imwrite(img_path, frame)
            captured_images.append(img_path)

            print(f"Captured frame {len(all_charuco_corners)} with {num_corners} corners")
        else:
            print("Not captured: not enough ChArUco corners")

    elif key == ord("c"):
        if len(all_charuco_corners) < MIN_CAPTURES_TO_CALIBRATE:
            print(f"Need at least {MIN_CAPTURES_TO_CALIBRATE} captures before calibration")
            continue

        try:
            if CALIBRATION_MODEL == "standard":
                latest_rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_standard(image_size)
                map1, map2, new_camera_matrix = make_standard_undistort_maps(
                    camera_matrix,
                    dist_coeffs,
                    image_size
                )

                print("Standard calibration complete")
                print("RMS:", latest_rms)
                print("Camera matrix:")
                print(camera_matrix)
                print("Distortion:")
                print(dist_coeffs)

            elif CALIBRATION_MODEL == "fisheye":
                latest_rms, fisheye_K, fisheye_D, rvecs, tvecs = calibrate_fisheye(image_size)
                map1, map2, new_camera_matrix = make_fisheye_undistort_maps(
                    fisheye_K,
                    fisheye_D,
                    image_size
                )

                print("Fisheye calibration complete")
                print("RMS:", latest_rms)
                print("K:")
                print(fisheye_K)
                print("D:")
                print(fisheye_D)

            else:
                raise ValueError("CALIBRATION_MODEL must be 'standard' or 'fisheye'")

            calibrated = True

        except cv2.error as e:
            print("Calibration failed:")
            print(e)

    elif key == ord("u"):
        if calibrated:
            undistort_preview = not undistort_preview
            print(f"Undistort preview: {undistort_preview}")
        else:
            print("Calibrate first with C")

    elif key == ord("s"):
        if not calibrated:
            print("Nothing to save. Calibrate first with C")
            continue

        if CALIBRATION_MODEL == "standard":
            save_calibration(
                "standard",
                latest_rms,
                camera_matrix,
                dist_coeffs,
                image_size
            )
        else:
            save_calibration(
                "fisheye",
                latest_rms,
                fisheye_K,
                fisheye_D,
                image_size
            )

    elif key == ord("r"):
        all_charuco_corners.clear()
        all_charuco_ids.clear()
        captured_images.clear()
        calibrated = False
        undistort_preview = False
        latest_rms = None
        map1 = None
        map2 = None
        print("Reset captures and calibration")

cap.release()
cv2.destroyAllWindows()
