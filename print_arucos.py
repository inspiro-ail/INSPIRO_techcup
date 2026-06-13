from __future__ import annotations

import argparse
import cv2


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


def parse_args():
    parser = argparse.ArgumentParser(description="Print detected ArUco marker IDs.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument(
        "--dictionary",
        default="DICT_5X5_1000",
        choices=DICTIONARIES.keys(),
        help="Use the same dictionary as your generated markers.",
    )
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def create_detector(dictionary_name: str):
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


def main():
    args = parse_args()

    detector = create_detector(args.dictionary)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    last_seen_ids = set()

    print("Printing detected ArUco marker IDs.")
    print("Press Q or ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read camera frame")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(detector, gray)

            current_ids = set()

            if ids is not None and len(ids):
                current_ids = {int(marker_id[0]) for marker_id in ids}

                for marker_id in sorted(current_ids):
                    if marker_id not in last_seen_ids:
                        print(f"Detected marker ID: {marker_id}")

                if not args.no_window:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    for marker_corners, marker_id_array in zip(corners, ids):
                        marker_id = int(marker_id_array[0])
                        points = marker_corners.reshape(4, 2)
                        x = int(points[:, 0].mean())
                        y = int(points[:, 1].mean())

                        cv2.putText(
                            frame,
                            f"ID {marker_id}",
                            (x - 35, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

            last_seen_ids = current_ids

            if not args.no_window:
                cv2.imshow("ArUco ID Printer", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()