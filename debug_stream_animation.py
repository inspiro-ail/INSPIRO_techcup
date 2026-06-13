"""
Debug streamer for a single animation text file.

By default this streams:
    animations/animation_export_20260426_162757.txt

Each line is written once to /dev/ttyUSB0 at 25 FPS and printed before write.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import serial


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ANIMATION = BASE_DIR / "animations" / "animation_export_20260426_162757.txt"
DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD_RATE = 115200
DEFAULT_FPS = 25.0
DEFAULT_STARTUP_DELAY = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream one animation txt file with verbose serial output.")
    parser.add_argument(
        "animation",
        nargs="?",
        default=str(DEFAULT_ANIMATION),
        help=f"Animation txt file to stream. Default: {DEFAULT_ANIMATION}",
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"Serial port. Default: {DEFAULT_PORT}")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE, help=f"Baud rate. Default: {DEFAULT_BAUD_RATE}")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help=f"Frames per second. Default: {DEFAULT_FPS:g}")
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=DEFAULT_STARTUP_DELAY,
        help=f"Seconds to wait after opening serial before frame 1. Default: {DEFAULT_STARTUP_DELAY:g}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    animation_path = Path(args.animation).expanduser()
    if not animation_path.is_absolute():
        animation_path = BASE_DIR / animation_path

    if not animation_path.exists():
        raise FileNotFoundError(f"Animation file not found: {animation_path}")

    frames = animation_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not frames:
        raise ValueError(f"Animation file has no frames: {animation_path}")

    period = 1.0 / args.fps
    print(f"Streaming {animation_path}")
    print(f"Frames: {len(frames)}")
    print(f"Serial: {args.port} @ {args.baud}")
    print(f"Rate: {args.fps:g} fps")
    print(f"Startup delay: {args.startup_delay:g}s")

    with serial.Serial(args.port, args.baud, timeout=0.1, write_timeout=1) as ser:
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass

        if args.startup_delay > 0:
            print(f"Opened serial. Waiting {args.startup_delay:g}s before frame 1...", flush=True)
            time.sleep(args.startup_delay)

        for index, frame in enumerate(frames, start=1):
            payload = f"{frame}\n".encode("utf-8")
            print(f"{index:04d}/{len(frames):04d}: {frame}", flush=True)
            ser.write(payload)
            time.sleep(period)

    print("Done.")


if __name__ == "__main__":
    main()
