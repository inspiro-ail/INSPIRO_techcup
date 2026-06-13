"""
Raspberry Pi web controller for a triangular holonomic kiwi drive.
hey
Requirements:
    pip install -r requirements.txt

Run:
    python control.py

Serial protocol matches the original pygame script:
    <Vx,Vy,Om>\n
"""

from __future__ import annotations

import atexit
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    import serial
except ImportError:  # Allows the web UI to run for development without pyserial.
    serial = None


BASE_DIR = Path(__file__).resolve().parent
ANIMATION_DIR = BASE_DIR / "animations"
AUDIO_DIR = BASE_DIR / "audio"
PROJECT_DIR = BASE_DIR / "projects"
SCRIPT_DIR = BASE_DIR / "scripts"
SERIAL_PORT = os.environ.get("KIWI_SERIAL_PORT", "/dev/ttyAMA0")
BAUD_RATE = int(os.environ.get("KIWI_BAUD_RATE", "115200"))
TARGET_HZ = 20
DEADBAND = 0.05
COMMAND_TIMEOUT_SECONDS = 0.45
VX_SIGN = float(os.environ.get("KIWI_VX_SIGN", "-1"))
VY_SIGN = float(os.environ.get("KIWI_VY_SIGN", "1"))
OM_SIGN = float(os.environ.get("KIWI_OM_SIGN", "1"))
ANIMATION_SERIAL_PORT = os.environ.get("KIWI_ANIMATION_SERIAL_PORT", "/dev/ttyUSB0")
ANIMATION_BAUD_RATE = int(os.environ.get("KIWI_ANIMATION_BAUD_RATE", "115200"))
ANIMATION_FPS = float(os.environ.get("KIWI_ANIMATION_FPS", "25"))
ANIMATION_START_DELAY = float(os.environ.get("KIWI_ANIMATION_START_DELAY", "1.25"))
PROJECT_ZERO_RETURN_SECONDS = float(os.environ.get("KIWI_PROJECT_ZERO_RETURN_SECONDS", "2.0"))
PROJECT_LIMIT_SECONDS = float(os.environ.get("KIWI_PROJECT_LIMIT_SECONDS", "180"))
AUDIO_PLAYER_COMMAND = os.environ.get("KIWI_AUDIO_PLAYER", "")
AUDIO_VOLUME_DEFAULT = float(os.environ.get("KIWI_AUDIO_VOLUME_DEFAULT", "10.0"))
AUDIO_VOLUME_MAX = float(os.environ.get("KIWI_AUDIO_VOLUME_MAX", "20.0"))
PROP_UDP_HOST = os.environ.get("KIWI_PROP_UDP_HOST", "255.255.255.255")
PROP_UDP_PORT = int(os.environ.get("KIWI_PROP_UDP_PORT", "4210"))
PROP_UDP_HZ = float(os.environ.get("KIWI_PROP_UDP_HZ", "10.0"))
SHUTDOWN_COMMAND = os.environ.get("KIWI_SHUTDOWN_COMMAND", "sudo shutdown -h now")
ALLOWED_ANIMATION_EXTENSIONS = {".txt"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
ALLOWED_SCRIPT_EXTENSIONS = {".py"}


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def cubic_bezier_ease(progress: float, x1: float, y1: float, x2: float, y2: float) -> float:
    progress = clamp(progress, 0.0, 1.0)
    x1 = clamp(x1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)

    def sample(axis1: float, axis2: float, t: float) -> float:
        inv = 1.0 - t
        return (3.0 * inv * inv * t * axis1) + (3.0 * inv * t * t * axis2) + (t * t * t)

    low = 0.0
    high = 1.0
    t = progress
    for _ in range(16):
        t = (low + high) * 0.5
        if sample(x1, x2, t) < progress:
            low = t
        else:
            high = t
    return sample(y1, y2, t)


def deadband(value: float) -> float:
    return 0.0 if abs(value) < DEADBAND else value


def is_animation_marker_line(line: str) -> bool:
    return "CODEWORD" in line.strip().upper().split()


def parse_animation_lines(lines: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    frames: list[str] = []
    markers: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if is_animation_marker_line(line):
            markers.append(
                {
                    "name": "CODEWORD",
                    "line": line_number,
                    "frame": len(frames),
                    "time": len(frames) / ANIMATION_FPS,
                }
            )
            continue
        frames.append(line)
    return frames, markers


def parse_animation_file(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    return parse_animation_lines(path.read_text(encoding="utf-8", errors="replace").splitlines())


def parse_joint_frame(line: str) -> list[float] | None:
    try:
        values = [float(value) for value in line.replace(",", " ").split()]
    except ValueError:
        return None
    return values or None


def format_joint_frame(values: list[float]) -> str:
    formatted = [f"{value:.2f}" for value in values]
    if len(formatted) == 18:
        return f"{','.join(formatted[:13])} {','.join(formatted[13:])}"
    return ",".join(formatted)


def normalize_command(payload: dict[str, Any]) -> tuple[float, float, float]:
    # The web client sends joystick intent. These signs map intent into the
    # robot firmware's serial coordinate system.
    vx = deadband(clamp(float(payload.get("vx", 0.0)) * VX_SIGN))
    vy = deadband(clamp(float(payload.get("vy", 0.0)) * VY_SIGN))
    om = deadband(clamp(float(payload.get("om", 0.0)) * OM_SIGN))
    return vx, vy, om


def encode_command(command: tuple[float, float, float]) -> bytes:
    vx, vy, om = command
    return f"<{vx:.3f},{vy:.3f},{om:.3f}>\n".encode("ascii")


class KiwiSerialStreamer:
    def __init__(self, port: str, baud_rate: int, target_hz: int) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.period = 1.0 / target_hz
        self.command = (0.0, 0.0, 0.0)
        self.last_command_time = 0.0
        self.last_sent = "<0.000,0.000,0.000>"
        self.error = ""
        self.suspended = False
        self.suspended_by = ""
        self.command_owner = ""
        self._serial = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_command(self, command: tuple[float, float, float], owner: str = "") -> bool:
        with self._lock:
            if self.command_owner and owner != self.command_owner:
                return False
            self.command = command
            self.last_command_time = time.monotonic()
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            command = self.command
            age = time.monotonic() - self.last_command_time if self.last_command_time else None
            suspended = self.suspended
            suspended_by = self.suspended_by
            command_owner = self.command_owner

        return {
            "port": self.port,
            "baudRate": self.baud_rate,
            "axisSigns": {"vx": VX_SIGN, "vy": VY_SIGN, "om": OM_SIGN},
            "connected": self.connected,
            "suspended": suspended,
            "suspendedBy": suspended_by,
            "commandOwner": command_owner,
            "command": {"vx": command[0], "vy": command[1], "om": command[2]},
            "commandAgeSeconds": age,
            "lastSent": self.last_sent,
            "error": self.error,
        }

    def acquire_control(self, owner: str) -> None:
        with self._lock:
            self.command_owner = owner
            self.command = (0.0, 0.0, 0.0)
            self.last_command_time = time.monotonic()

    def release_control(self, owner: str) -> None:
        with self._lock:
            if self.command_owner == owner:
                self.command_owner = ""
                self.command = (0.0, 0.0, 0.0)
                self.last_command_time = time.monotonic()

    def suspend(self, owner: str) -> None:
        with self._lock:
            self.suspended = True
            self.suspended_by = owner

    def resume(self, owner: str) -> None:
        with self._lock:
            if self.suspended_by in {"", owner}:
                self.suspended = False
                self.suspended_by = ""
                self.command = (0.0, 0.0, 0.0)
                self.last_command_time = 0.0

    @property
    def connected(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._write_stop()
        if self._serial and self._serial.is_open:
            self._serial.close()

    def _connect(self) -> None:
        if serial is None:
            self.error = "pyserial is not installed"
            return

        try:
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            self.error = ""
            print(f"Connected to {self.port} @ {self.baud_rate}")
        except Exception as exc:  # Keep the web app alive so status is visible.
            self._serial = None
            self.error = f"Could not open {self.port}: {exc}"

    def _current_command(self) -> tuple[float, float, float]:
        with self._lock:
            if self.suspended:
                return self.command
            stale = time.monotonic() - self.last_command_time > COMMAND_TIMEOUT_SECONDS
            return (0.0, 0.0, 0.0) if stale else self.command

    def _write(self, command: tuple[float, float, float]) -> None:
        if not self.connected:
            return

        frame = encode_command(command)
        self._serial.write(frame)
        self.last_sent = frame.decode("ascii").strip()

    def _write_stop(self) -> None:
        try:
            for _ in range(3):
                self._write((0.0, 0.0, 0.0))
                time.sleep(0.02)
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self.connected:
                self._connect()
                time.sleep(1.0 if not self.connected else 0.0)
                continue

            try:
                with self._lock:
                    suspended = self.suspended
                if not suspended:
                    self._write(self._current_command())
            except Exception as exc:
                self.error = f"Serial write error: {exc}"
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None

            time.sleep(self.period)


class AnimationPlayer:
    def __init__(self, directory: Path, port: str, baud_rate: int, fps: float, start_delay: float) -> None:
        self.directory = directory
        self.port = port
        self.baud_rate = baud_rate
        self.period = 1.0 / fps
        self.fps = fps
        self.start_delay = start_delay
        self.current_file = ""
        self.frame_index = 0
        self.frame_count = 0
        self.last_sent = ""
        self.error = ""
        self.playing = False
        self.phase = "idle"
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_frame_values: list[float] | None = None
        self.directory.mkdir(exist_ok=True)

    def list_files(self) -> list[dict[str, Any]]:
        files = []
        for path in sorted(self.directory.glob("*.txt")):
            if path.is_file():
                frames, markers = parse_animation_file(path)
                files.append(
                    {
                        "name": path.name,
                        "size": path.stat().st_size,
                        "frames": len(frames),
                        "markers": markers,
                    }
                )
        return files

    def save_upload(self, uploaded_file) -> str:
        filename = secure_filename(uploaded_file.filename or "")
        if not filename:
            raise ValueError("Missing file name")

        path = self._safe_path(filename)
        if path.suffix.lower() not in ALLOWED_ANIMATION_EXTENSIONS:
            raise ValueError("Only .txt animation files are supported")

        uploaded_file.save(path)
        return path.name

    def play(self, filename: str) -> None:
        path = self._safe_path(filename)
        if not path.exists() or not path.is_file():
            raise ValueError("Animation file does not exist")

        with self._lock:
            if self.playing:
                raise RuntimeError("An animation is already playing")

            frames, markers = parse_animation_file(path)
            if not frames:
                raise ValueError("Animation file has no frames")

            self._cancel.clear()
            self.current_file = path.name
            self.frame_index = 0
            self.frame_count = len(frames)
            self.last_sent = f"{len(markers)} marker(s)" if markers else ""
            self.error = ""
            self._last_frame_values = None
            self.playing = True
            self.phase = "opening"
            self._thread = threading.Thread(target=self._run, args=(path.name, frames), daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._cancel.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "port": self.port,
                "baudRate": self.baud_rate,
                "fps": self.fps,
                "startupDelaySeconds": self.start_delay,
                "playing": self.playing,
                "phase": self.phase,
                "currentFile": self.current_file,
                "frameIndex": self.frame_index,
                "frameCount": self.frame_count,
                "lastSent": self.last_sent,
                "error": self.error,
            }

    def _safe_path(self, filename: str) -> Path:
        name = secure_filename(filename)
        path = (self.directory / name).resolve()
        if self.directory.resolve() not in path.parents:
            raise ValueError("Invalid animation path")
        return path

    def _run(self, filename: str, frames: list[str]) -> None:
        try:
            if serial is None:
                raise RuntimeError("pyserial is not installed")

            with serial.Serial(self.port, self.baud_rate, timeout=0.1, write_timeout=1) as ser:
                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception:
                    pass

                with self._lock:
                    self.phase = "arming"
                    self.last_sent = f"Waiting {self.start_delay:g}s for serial device"

                deadline = time.monotonic() + self.start_delay
                while time.monotonic() < deadline:
                    if self._cancel.is_set():
                        return
                    time.sleep(0.02)

                with self._lock:
                    self.phase = "playing"
                    self.last_sent = ""

                last_frame = ""
                for index, frame in enumerate(frames, start=1):
                    if self._cancel.is_set():
                        break

                    if frame.strip():
                        last_frame = frame
                    if not last_frame:
                        time.sleep(self.period)
                        continue
                    frame_to_send = last_frame
                    payload = f"{frame_to_send}\n".encode("utf-8")
                    ser.write(payload)
                    values = parse_joint_frame(frame_to_send)
                    if values is not None:
                        self._last_frame_values = values
                    with self._lock:
                        self.frame_index = index
                        self.last_sent = frame_to_send
                    time.sleep(self.period)
                self._return_to_zero(ser)
        except Exception as exc:
            with self._lock:
                self.error = f"Animation playback error: {exc}"
        finally:
            with self._lock:
                self.playing = False
                self.phase = "idle"
                self.current_file = filename

    def _return_to_zero(self, ser) -> None:
        start_values = self._last_frame_values
        if not start_values or PROJECT_ZERO_RETURN_SECONDS <= 0.0:
            return
        if max(abs(value) for value in start_values) < 0.01:
            return

        with self._lock:
            self.phase = "zeroing"
            self.last_sent = "Returning joints to zero"

        steps = max(1, int(PROJECT_ZERO_RETURN_SECONDS * self.fps))
        for step in range(1, steps + 1):
            progress = step / steps
            eased = progress * progress * (3.0 - 2.0 * progress)
            frame_values = [value * (1.0 - eased) for value in start_values]
            frame = format_joint_frame(frame_values)
            ser.write(f"{frame}\n".encode("utf-8"))
            with self._lock:
                self.last_sent = frame
            time.sleep(self.period)

        zero_frame = format_joint_frame([0.0] * len(start_values))
        ser.write(f"{zero_frame}\n".encode("utf-8"))
        self._last_frame_values = [0.0] * len(start_values)
        with self._lock:
            self.last_sent = zero_frame


class PropBoolBroadcaster:
    DEFAULT_STATES = {0: True, 1: True, 2: True, 3: False}

    def __init__(self, host: str, port: int, hz: float) -> None:
        self.host = host
        self.port = port
        self.period = 1.0 / max(hz, 0.1)
        self.states = dict(self.DEFAULT_STATES)
        self.seq = 0
        self.last_payload = ""
        self.last_error = ""
        self.last_sent_at = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close()

    def reset_defaults(self) -> None:
        with self._lock:
            self.states = dict(self.DEFAULT_STATES)
        self.send_once()

    def flip(self, prop_id: int) -> None:
        if prop_id not in self.DEFAULT_STATES:
            raise ValueError("Prop ID must be 0, 1, 2, or 3")
        with self._lock:
            self.states[prop_id] = not self.states[prop_id]
        self.send_once()

    def set_state(self, prop_id: int, value: bool) -> None:
        if prop_id not in self.DEFAULT_STATES:
            raise ValueError("Prop ID must be 0, 1, 2, or 3")
        with self._lock:
            self.states[prop_id] = bool(value)
        self.send_once()

    def status(self) -> dict[str, Any]:
        with self._lock:
            states = dict(self.states)
            seq = self.seq
            last_payload = self.last_payload
            last_error = self.last_error
            last_sent_at = self.last_sent_at
        return {
            "host": self.host,
            "port": self.port,
            "states": states,
            "seq": seq,
            "lastPayload": last_payload,
            "lastError": last_error,
            "lastSentAt": last_sent_at,
        }

    def _payload(self) -> bytes:
        with self._lock:
            self.seq += 1
            packet = {
                "type": "prop_bool_state",
                "seq": self.seq,
                "timestamp": round(time.time(), 3),
                "states": {str(prop_id): bool(self.states[prop_id]) for prop_id in sorted(self.states)},
            }
            payload = json.dumps(packet, separators=(",", ":"))
            self.last_payload = payload
        return f"{payload}\n".encode("utf-8")

    def send_once(self) -> None:
        try:
            self._socket.sendto(self._payload(), (self.host, self.port))
            with self._lock:
                self.last_error = ""
                self.last_sent_at = time.time()
        except OSError as exc:
            with self._lock:
                self.last_error = str(exc)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.send_once()
            self._stop.wait(self.period)


def safe_child_path(directory: Path, filename: str, allowed_extensions: set[str] | None = None) -> Path:
    name = secure_filename(filename)
    if not name:
        raise ValueError("Missing file name")

    path = (directory / name).resolve()
    directory_resolved = directory.resolve()
    if path != directory_resolved and directory_resolved not in path.parents:
        raise ValueError("Invalid file path")

    if allowed_extensions is not None and path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return path


def count_animation_frames(filename: str) -> int:
    path = safe_child_path(ANIMATION_DIR, filename, ALLOWED_ANIMATION_EXTENSIONS)
    if not path.exists() or not path.is_file():
        raise ValueError("Animation file does not exist")
    frames, _markers = parse_animation_file(path)
    return len(frames)


def animation_markers(filename: str) -> list[dict[str, Any]]:
    path = safe_child_path(ANIMATION_DIR, filename, ALLOWED_ANIMATION_EXTENSIONS)
    if not path.exists() or not path.is_file():
        raise ValueError("Animation file does not exist")
    _frames, markers = parse_animation_file(path)
    return markers


def audio_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / rate if rate else None
        except Exception:
            return None

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return float(result.stdout.strip()) if result.stdout.strip() else None
    except Exception:
        return None


def list_audio_files() -> list[dict[str, Any]]:
    AUDIO_DIR.mkdir(exist_ok=True)
    files = []
    for path in sorted(AUDIO_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
            files.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "duration": audio_duration_seconds(path),
                }
            )
    return files


def list_script_files() -> list[dict[str, Any]]:
    SCRIPT_DIR.mkdir(exist_ok=True)
    files = []
    for path in sorted(SCRIPT_DIR.glob("*.py")):
        if path.is_file():
            files.append({"name": path.name, "size": path.stat().st_size})
    return files


def list_project_files() -> list[dict[str, Any]]:
    PROJECT_DIR.mkdir(exist_ok=True)
    files = []
    for path in sorted(PROJECT_DIR.glob("*.json")):
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            files.append(
                {
                    "name": path.name,
                    "title": data.get("title") or path.stem,
                    "duration": float(data.get("duration", 0) or 0),
                    "blocks": len(data.get("blocks", []) or []),
                }
            )
    return files


def project_safe_path(filename: str) -> Path:
    name = secure_filename(filename)
    if not name:
        raise ValueError("Missing project name")
    if not name.endswith(".json"):
        name = f"{name}.json"
    return safe_child_path(PROJECT_DIR, name, {".json"})


def normalize_audio_volume(value: Any) -> float:
    try:
        return clamp(float(value), 0.0, AUDIO_VOLUME_MAX)
    except (TypeError, ValueError):
        return clamp(AUDIO_VOLUME_DEFAULT, 0.0, AUDIO_VOLUME_MAX)


def ffplay_audio_command(path: Path, volume: float, start_seconds: float = 0.0) -> list[str] | None:
    executable = shutil.which("ffplay")
    if not executable:
        return None

    command = [
        executable,
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "quiet",
    ]
    if start_seconds > 0.0:
        command.extend(["-ss", f"{start_seconds:.3f}"])
    command.extend(["-af", f"volume={volume:g}", str(path)])
    return command


def choose_audio_command(path: Path, volume: float = AUDIO_VOLUME_DEFAULT, start_seconds: float = 0.0) -> list[str]:
    volume = normalize_audio_volume(volume)
    start_seconds = max(0.0, float(start_seconds or 0.0))
    command = AUDIO_PLAYER_COMMAND.strip()
    if not command:
        if start_seconds > 0.0:
            ffplay_command = ffplay_audio_command(path, volume, start_seconds)
            if ffplay_command:
                return ffplay_command
            raise RuntimeError("Negative audio offset clipping requires ffplay or a KIWI_AUDIO_PLAYER with {start}.")

        for candidate in ("mpg123", "ffplay", "paplay", "aplay"):
            executable = shutil.which(candidate)
            if executable:
                if candidate == "mpg123":
                    return [executable, "-q", "-f", str(max(0, int(32768 * volume))), str(path)]
                if candidate == "ffplay":
                    return ffplay_audio_command(path, volume) or [executable, "-nodisp", "-autoexit", str(path)]
                if candidate == "paplay":
                    return [executable, f"--volume={max(0, int(65536 * volume))}", str(path)]
                return [executable, str(path)]
        raise RuntimeError("No audio player found. Install mpg123/ffplay/aplay or set KIWI_AUDIO_PLAYER.")

    parts = shlex.split(command)
    parts = [part.replace("{volume}", f"{volume:g}") for part in parts]
    parts = [part.replace("{start}", f"{start_seconds:.3f}") for part in parts]
    if start_seconds > 0.0 and "{start}" not in command:
        raise RuntimeError("KIWI_AUDIO_PLAYER must include {start} to support negative audio offset clipping.")
    if any("{file}" in part for part in parts):
        return [part.replace("{file}", str(path)) for part in parts]
    return [*parts, str(path)]


class ProjectPlayer:
    def __init__(self) -> None:
        self.playing = False
        self.paused = False
        self.current_project = ""
        self.phase = "idle"
        self.started_at = 0.0
        self.pause_started_at = 0.0
        self.pause_offset = 0.0
        self.duration = 0.0
        self.playhead = 0.0
        self.error = ""
        self.active_blocks: list[str] = []
        self._project: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._animation_lock = threading.Lock()
        self._animation_serial_lock = threading.Lock()
        self._animation_serial = None
        self._last_animation_frame_values: list[float] | None = None
        self._processes: dict[str, subprocess.Popen] = {}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "playing": self.playing,
                "paused": self.paused,
                "phase": self.phase,
                "project": self.current_project,
                "duration": self.duration,
                "playhead": self.playhead,
                "activeBlocks": list(self.active_blocks),
                "error": self.error,
            }

    def save_audio_upload(self, uploaded_file) -> str:
        filename = secure_filename(uploaded_file.filename or "")
        path = safe_child_path(AUDIO_DIR, filename, ALLOWED_AUDIO_EXTENSIONS)
        AUDIO_DIR.mkdir(exist_ok=True)
        uploaded_file.save(path)
        return path.name

    def validate_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        title = str(payload.get("title") or "Untitled Project").strip()[:80] or "Untitled Project"
        raw_blocks = payload.get("blocks") or []
        if not isinstance(raw_blocks, list):
            raise ValueError("Project blocks must be a list")

        blocks: list[dict[str, Any]] = []
        for index, raw_block in enumerate(raw_blocks):
            if not isinstance(raw_block, dict):
                raise ValueError("Each block must be an object")

            block_type = str(raw_block.get("type") or "").strip()
            if block_type not in {"animation", "audio", "python", "wheelbase", "prop_flip", "prop_pulse"}:
                raise ValueError(f"Unsupported block type: {block_type}")

            start = max(0.0, float(raw_block.get("start", 0) or 0))
            channel = int(raw_block.get("channel", 0) or 0)
            if channel < 0 or channel > 2:
                raise ValueError("Channel must be 0, 1, or 2")

            block = {
                "id": str(raw_block.get("id") or f"block-{index + 1}"),
                "type": block_type,
                "channel": channel,
                "start": start,
                "title": str(raw_block.get("title") or block_type).strip()[:80],
                "blocking": bool(raw_block.get("blocking", False)),
            }

            duration = float(raw_block.get("duration", 0) or 0)
            if block_type == "animation":
                animation = str(raw_block.get("animation") or "")
                animation_frames = count_animation_frames(animation)
                block["animation"] = animation
                block["frames"] = animation_frames
                block["markers"] = animation_markers(animation)
                animation_duration = animation_frames / ANIMATION_FPS
                duration = animation_duration
                audio = str(raw_block.get("audio") or "")
                if audio:
                    audio_offset = float(raw_block.get("audioOffset", 0.0) or 0.0)
                    audio_path = safe_child_path(AUDIO_DIR, audio, ALLOWED_AUDIO_EXTENSIONS)
                    if not audio_path.exists() or not audio_path.is_file():
                        raise ValueError("Audio file does not exist")
                    block["audio"] = audio
                    block["audioOffset"] = audio_offset
                    block["volume"] = normalize_audio_volume(raw_block.get("volume", AUDIO_VOLUME_DEFAULT))
            elif block_type == "audio":
                audio = str(raw_block.get("audio") or "")
                audio_path = safe_child_path(AUDIO_DIR, audio, ALLOWED_AUDIO_EXTENSIONS)
                if not audio_path.exists() or not audio_path.is_file():
                    raise ValueError("Audio file does not exist")
                block["audio"] = audio
                block["volume"] = normalize_audio_volume(raw_block.get("volume", AUDIO_VOLUME_DEFAULT))
                duration = audio_duration_seconds(audio_path) or duration or 5.0
            elif block_type == "python":
                script = str(raw_block.get("script") or "")
                script_path = safe_child_path(SCRIPT_DIR, script, ALLOWED_SCRIPT_EXTENSIONS)
                if not script_path.exists() or not script_path.is_file():
                    raise ValueError("Script file does not exist")
                block["script"] = script
                block["blocking"] = bool(raw_block.get("blocking", True))
                duration = max(0.1, duration or 5.0)
            elif block_type == "wheelbase":
                direction = str(raw_block.get("direction") or "left").strip().lower()
                if direction not in {"left", "right"}:
                    raise ValueError("Wheelbase direction must be left or right")
                min_speed = clamp(float(raw_block.get("minSpeed", 0.15) or 0.0), 0.0, 1.0)
                max_speed = clamp(float(raw_block.get("maxSpeed", 0.45) or 0.0), 0.0, 1.0)
                if min_speed > max_speed:
                    min_speed, max_speed = max_speed, min_speed
                bezier = raw_block.get("bezier") if isinstance(raw_block.get("bezier"), dict) else {}
                block["direction"] = direction
                block["minSpeed"] = min_speed
                block["maxSpeed"] = max_speed
                block["bezier"] = {
                    "x1": clamp(float(bezier.get("x1", 0.42) or 0.0), 0.0, 1.0),
                    "y1": clamp(float(bezier.get("y1", 0.0) or 0.0), 0.0, 1.0),
                    "x2": clamp(float(bezier.get("x2", 0.58) or 0.0), 0.0, 1.0),
                    "y2": clamp(float(bezier.get("y2", 1.0) or 0.0), 0.0, 1.0),
                }
                block["title"] = f"Rotate {direction}"
                duration = max(0.1, duration or 1.0)
            elif block_type == "prop_flip":
                prop_id = int(raw_block.get("propId", 0) or 0)
                if prop_id not in {0, 1, 2}:
                    raise ValueError("Prop flip block supports prop IDs 0, 1, and 2")
                block["propId"] = prop_id
                block["title"] = f"Flip prop {prop_id}"
                duration = 0.1
            elif block_type == "prop_pulse":
                block["propId"] = 3
                block["title"] = "Pulse prop 3"
                duration = max(0.1, duration or 1.0)

            block["duration"] = max(0.1, duration)
            if block["start"] + block["duration"] > PROJECT_LIMIT_SECONDS:
                raise ValueError(f"Project exceeds {PROJECT_LIMIT_SECONDS:g}s limit")
            blocks.append(block)

        blocks.sort(key=lambda item: (item["start"], item["channel"]))
        self._validate_overlaps(blocks)
        duration = max((block["start"] + block["duration"] for block in blocks), default=0.0)
        return {
            "title": title,
            "duration": duration,
            "limit": PROJECT_LIMIT_SECONDS,
            "channels": 3,
            "blocks": blocks,
        }

    def _validate_overlaps(self, blocks: list[dict[str, Any]]) -> None:
        for index, left in enumerate(blocks):
            left_start = left["start"]
            left_end = left_start + left["duration"]
            for right in blocks[index + 1 :]:
                right_start = right["start"]
                right_end = right_start + right["duration"]
                overlap = left_start < right_end and right_start < left_end
                if not overlap:
                    continue
                if left["channel"] == right["channel"]:
                    raise ValueError("Blocks cannot overlap on the same channel")
                if left["type"] == "animation" and right["type"] == "animation":
                    raise ValueError("Animation blocks cannot overlap because /dev/ttyUSB0 is shared")
                if left["type"] in {"python", "wheelbase"} and right["type"] in {"python", "wheelbase"}:
                    raise ValueError("Python and wheelbase blocks cannot overlap because the drive UART is shared")

    def save_project(self, filename: str, payload: dict[str, Any]) -> dict[str, Any]:
        project = self.validate_project(payload)
        path = project_safe_path(filename or project["title"])
        PROJECT_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(project, indent=2), encoding="utf-8")
        return {"filename": path.name, "project": project}

    def load_project(self, filename: str) -> dict[str, Any]:
        path = project_safe_path(filename)
        if not path.exists() or not path.is_file():
            raise ValueError("Project file does not exist")
        return self.validate_project(json.loads(path.read_text(encoding="utf-8")))

    def play(self, filename: str) -> None:
        project = self.load_project(filename)
        with self._lock:
            if self.playing:
                raise RuntimeError("A project is already playing")
            self.playing = True
            self.paused = False
            self.current_project = filename
            self.phase = "playing"
            self.duration = float(project.get("duration", 0) or 0)
            self.playhead = 0.0
            self.error = ""
            self.active_blocks = []
            self.pause_offset = 0.0
            self.pause_started_at = 0.0
            self._project = project
            self._last_animation_frame_values = None
            prop_broadcaster.reset_defaults()
            self._cancel.clear()
            self._paused.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if not self.playing or self.paused:
                return
            self.paused = True
            self.phase = "paused"
            self.pause_started_at = time.monotonic()
            self._paused.set()
        self._pause_processes(True)

    def resume(self) -> None:
        with self._lock:
            if not self.playing or not self.paused:
                return
            self.pause_offset += time.monotonic() - self.pause_started_at
            self.paused = False
            self.phase = "playing"
            self.pause_started_at = 0.0
            self._paused.clear()
        self._pause_processes(False)

    def stop(self) -> None:
        self._cancel.set()
        self._paused.clear()
        for process in list(self._processes.values()):
            if process.poll() is None:
                process.terminate()
        with self._lock:
            self.paused = False
            self.phase = "stopping"
            self.active_blocks = []

    def _timeline_time(self) -> float:
        paused_extra = time.monotonic() - self.pause_started_at if self.paused and self.pause_started_at else 0.0
        return max(0.0, time.monotonic() - self.started_at - self.pause_offset - paused_extra)

    def _run(self) -> None:
        assert self._project is not None
        blocks = self._runtime_blocks(list(self._project.get("blocks", [])))
        self.started_at = time.monotonic()
        threads: list[threading.Thread] = []
        self._close_animation_serial()
        self._last_animation_frame_values = None

        try:
            for block in blocks:
                while not self._cancel.is_set():
                    self._wait_if_paused()
                    playhead = self._timeline_time()
                    with self._lock:
                        self.playhead = min(playhead, self.duration)
                    if playhead >= block["start"]:
                        break
                    time.sleep(0.01)

                if self._cancel.is_set():
                    break

                if block["type"] == "python" and block.get("blocking"):
                    started = time.monotonic()
                    pause_offset_started = self.pause_offset
                    self._run_python_block(block, wait=True)
                    elapsed = time.monotonic() - started
                    pause_elapsed = self.pause_offset - pause_offset_started
                    self.pause_offset += max(0.0, elapsed - pause_elapsed)
                    continue

                thread = threading.Thread(target=self._run_block, args=(block,), daemon=True)
                thread.start()
                threads.append(thread)

            while not self._cancel.is_set() and any(thread.is_alive() for thread in threads):
                self._wait_if_paused()
                with self._lock:
                    self.playhead = min(self._timeline_time(), self.duration)
                time.sleep(0.03)
        except Exception as exc:
            with self._lock:
                self.error = f"Project playback error: {exc}"
        finally:
            if self._cancel.is_set():
                for thread in threads:
                    thread.join(timeout=1.0)
            self._return_animation_to_zero(ignore_cancel=True)
            prop_broadcaster.reset_defaults()
            self._close_animation_serial()
            with self._lock:
                self.playhead = self.duration if not self._cancel.is_set() else self.playhead
                self.playing = False
                self.paused = False
                self.phase = "idle"
                self.active_blocks = []

    def _run_block(self, block: dict[str, Any]) -> None:
        block_id = str(block["id"])
        self._mark_block(block_id, True)
        try:
            if block["type"] == "animation":
                self._run_animation_block(block)
            elif block["type"] == "audio":
                process = self._start_audio(block["audio"], block_id, block.get("volume", AUDIO_VOLUME_DEFAULT))
                process.wait()
            elif block["type"] == "python":
                self._run_python_block(block, wait=False)
            elif block["type"] == "wheelbase":
                self._run_wheelbase_block(block)
            elif block["type"] == "prop_flip":
                self._run_prop_flip_block(block)
            elif block["type"] == "prop_pulse":
                self._run_prop_pulse_block(block)
        except Exception as exc:
            with self._lock:
                self.error = f"Block {block_id} failed: {exc}"
        finally:
            self._mark_block(block_id, False)

    def _runtime_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(blocks, key=lambda item: (item["start"], item.get("channel", 0)))

    def _run_animation_block(self, block: dict[str, Any]) -> None:
        path = safe_child_path(ANIMATION_DIR, block["animation"], ALLOWED_ANIMATION_EXTENSIONS)
        frames, _markers = parse_animation_file(path)
        if not frames:
            raise ValueError("Animation file has no frames")
        if serial is None:
            raise RuntimeError("pyserial is not installed")

        with self._animation_lock:
            ser, opened_now = self._open_animation_serial()
            if opened_now:
                self._sleep_project_time(ANIMATION_START_DELAY)

            audio = block.get("audio")
            elapsed = 0.0
            if audio:
                audio_offset = float(block.get("audioOffset", 0.0) or 0.0)
                if audio_offset <= 0.0:
                    self._start_audio(
                        audio,
                        f"{block['id']}:audio",
                        block.get("volume", AUDIO_VOLUME_DEFAULT),
                        start_seconds=abs(audio_offset),
                    )
                else:
                    threading.Thread(
                        target=self._start_delayed_audio,
                        args=(audio, f"{block['id']}:audio", block.get("volume", AUDIO_VOLUME_DEFAULT), audio_offset),
                        daemon=True,
                    ).start()

            last_frame = ""
            for frame in frames:
                if self._cancel.is_set():
                    break
                self._wait_if_paused()
                if frame.strip():
                    last_frame = frame
                if not last_frame:
                    self._sleep_project_time(1.0 / ANIMATION_FPS)
                    elapsed += 1.0 / ANIMATION_FPS
                    continue
                ser.write(f"{last_frame}\n".encode("utf-8"))
                values = parse_joint_frame(last_frame)
                if values is not None:
                    self._last_animation_frame_values = values
                self._sleep_project_time(1.0 / ANIMATION_FPS)
                elapsed += 1.0 / ANIMATION_FPS

            remaining = float(block.get("duration", elapsed) or elapsed) - elapsed
            if remaining > 0.0:
                self._sleep_project_time(remaining)

    def _start_delayed_audio(self, filename: str, owner: str, volume: float, delay: float) -> None:
        self._sleep_project_time(delay)
        if not self._cancel.is_set():
            self._start_audio(filename, owner, volume)

    def _return_animation_to_zero(self, ignore_cancel: bool = False) -> None:
        start_values = self._last_animation_frame_values
        if not start_values or PROJECT_ZERO_RETURN_SECONDS <= 0.0:
            return

        if max(abs(value) for value in start_values) < 0.01:
            return

        try:
            with self._lock:
                self.phase = "zeroing"
            with self._animation_lock:
                ser, _opened_now = self._open_animation_serial()
                steps = max(1, int(PROJECT_ZERO_RETURN_SECONDS * ANIMATION_FPS))
                period = 1.0 / ANIMATION_FPS
                for step in range(1, steps + 1):
                    if self._cancel.is_set() and not ignore_cancel:
                        break
                    progress = step / steps
                    eased = progress * progress * (3.0 - 2.0 * progress)
                    frame_values = [value * (1.0 - eased) for value in start_values]
                    ser.write(f"{format_joint_frame(frame_values)}\n".encode("utf-8"))
                    if ignore_cancel:
                        time.sleep(period)
                    else:
                        self._sleep_project_time(period)
                ser.write(f"{format_joint_frame([0.0] * len(start_values))}\n".encode("utf-8"))
                self._last_animation_frame_values = [0.0] * len(start_values)
        except Exception as exc:
            with self._lock:
                self.error = f"Joint zero return failed: {exc}"

    def _run_wheelbase_block(self, block: dict[str, Any]) -> None:
        block_id = str(block["id"])
        owner = f"project-wheelbase:{block_id}"
        duration = max(0.1, float(block.get("duration", 1.0) or 1.0))
        min_speed = clamp(float(block.get("minSpeed", 0.15) or 0.0), 0.0, 1.0)
        max_speed = clamp(float(block.get("maxSpeed", 0.45) or 0.0), 0.0, 1.0)
        if min_speed > max_speed:
            min_speed, max_speed = max_speed, min_speed
        direction = -1.0 if block.get("direction") == "left" else 1.0
        bezier = block.get("bezier") if isinstance(block.get("bezier"), dict) else {}
        x1 = float(bezier.get("x1", 0.42) or 0.0)
        y1 = float(bezier.get("y1", 0.0) or 0.0)
        x2 = float(bezier.get("x2", 0.58) or 0.0)
        y2 = float(bezier.get("y2", 1.0) or 0.0)
        period = 1.0 / TARGET_HZ
        elapsed = 0.0

        streamer.acquire_control(owner)
        try:
            while not self._cancel.is_set() and elapsed < duration:
                self._wait_if_paused()
                progress = clamp(elapsed / duration, 0.0, 1.0)
                ramp_progress = progress * 2.0 if progress <= 0.5 else (1.0 - progress) * 2.0
                eased = cubic_bezier_ease(ramp_progress, x1, y1, x2, y2)
                speed = min_speed + (max_speed - min_speed) * eased
                om = clamp(direction * speed * OM_SIGN)
                streamer.set_command((0.0, 0.0, om), owner=owner)
                step = min(period, duration - elapsed)
                self._sleep_project_time(step)
                elapsed += step
        finally:
            streamer.set_command((0.0, 0.0, 0.0), owner=owner)
            streamer.release_control(owner)

    def _run_prop_flip_block(self, block: dict[str, Any]) -> None:
        prop_broadcaster.flip(int(block.get("propId", 0)))
        self._sleep_project_time(float(block.get("duration", 0.1) or 0.1))

    def _run_prop_pulse_block(self, block: dict[str, Any]) -> None:
        duration = max(0.1, float(block.get("duration", 1.0) or 1.0))
        prop_broadcaster.set_state(3, True)
        try:
            self._sleep_project_time(duration)
        finally:
            prop_broadcaster.set_state(3, False)

    def _open_animation_serial(self):
        with self._animation_serial_lock:
            if self._animation_serial is not None and getattr(self._animation_serial, "is_open", True):
                return self._animation_serial, False

            ser = serial.Serial(ANIMATION_SERIAL_PORT, ANIMATION_BAUD_RATE, timeout=0.1, write_timeout=1)
            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass
            self._animation_serial = ser
            return ser, True

    def _close_animation_serial(self) -> None:
        with self._animation_serial_lock:
            ser = self._animation_serial
            self._animation_serial = None

        if ser is None:
            return

        try:
            ser.close()
        except Exception:
            pass

    def _start_audio(
        self,
        filename: str,
        owner: str,
        volume: float = AUDIO_VOLUME_DEFAULT,
        start_seconds: float = 0.0,
    ) -> subprocess.Popen:
        path = safe_child_path(AUDIO_DIR, filename, ALLOWED_AUDIO_EXTENSIONS)
        command = choose_audio_command(path, volume, start_seconds)
        process = subprocess.Popen(command)
        self._processes[owner] = process
        return process

    def _run_python_block(self, block: dict[str, Any], wait: bool) -> None:
        block_id = str(block["id"])
        owner = f"project-python:{block_id}"
        if wait:
            self._mark_block(block_id, True)
        try:
            script = safe_child_path(SCRIPT_DIR, block["script"], ALLOWED_SCRIPT_EXTENSIONS)
            streamer.suspend(owner)
            process = subprocess.Popen([sys.executable, str(script)], cwd=str(BASE_DIR))
            self._processes[block_id] = process
            while process.poll() is None and not self._cancel.is_set():
                self._wait_if_paused()
                time.sleep(0.05)
            if self._cancel.is_set() and process.poll() is None:
                process.terminate()
        finally:
            streamer.resume(owner)
            self._processes.pop(block_id, None)
            if wait:
                self._mark_block(block_id, False)

    def _sleep_project_time(self, seconds: float) -> None:
        remaining = max(0.0, seconds)
        while not self._cancel.is_set() and remaining > 0.0:
            self._wait_if_paused()
            chunk = min(remaining, 0.01)
            time.sleep(chunk)
            remaining -= chunk

    def _wait_if_paused(self) -> None:
        while self._paused.is_set() and not self._cancel.is_set():
            time.sleep(0.03)

    def _mark_block(self, block_id: str, active: bool) -> None:
        with self._lock:
            if active and block_id not in self.active_blocks:
                self.active_blocks.append(block_id)
            elif not active and block_id in self.active_blocks:
                self.active_blocks.remove(block_id)

    def _pause_processes(self, paused: bool) -> None:
        stop_signal = getattr(signal, "SIGSTOP", None)
        continue_signal = getattr(signal, "SIGCONT", None)
        chosen_signal = stop_signal if paused else continue_signal
        if chosen_signal is None:
            return
        for process in list(self._processes.values()):
            if process.poll() is None:
                try:
                    os.kill(process.pid, chosen_signal)
                except Exception:
                    pass


for directory in (ANIMATION_DIR, AUDIO_DIR, PROJECT_DIR, SCRIPT_DIR):
    directory.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024
streamer = KiwiSerialStreamer(SERIAL_PORT, BAUD_RATE, TARGET_HZ)
animation_player = AnimationPlayer(
    ANIMATION_DIR,
    ANIMATION_SERIAL_PORT,
    ANIMATION_BAUD_RATE,
    ANIMATION_FPS,
    ANIMATION_START_DELAY,
)
prop_broadcaster = PropBoolBroadcaster(PROP_UDP_HOST, PROP_UDP_PORT, PROP_UDP_HZ)
project_player = ProjectPlayer()
streamer.start()
prop_broadcaster.start()
atexit.register(streamer.close)
atexit.register(prop_broadcaster.close)
atexit.register(animation_player.stop)
atexit.register(project_player.stop)


@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename: str):
    if filename not in {"app.js", "style.css"}:
        return ("Not found", 404)
    return send_from_directory(BASE_DIR, filename)


@app.get("/api/status")
def api_status():
    return jsonify(streamer.status())


@app.post("/api/drive")
def api_drive():
    try:
        command = normalize_command(request.get_json(force=True, silent=False) or {})
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid command payload: {exc}"}), 400

    accepted = streamer.set_command(command)
    if not accepted:
        status = streamer.status()
        active = status.get("command", {})
        return jsonify(
            {
                "ok": True,
                "accepted": False,
                "owner": status.get("commandOwner", ""),
                "vx": active.get("vx", 0.0),
                "vy": active.get("vy", 0.0),
                "om": active.get("om", 0.0),
            }
        )
    return jsonify({"ok": True, "accepted": True, "vx": command[0], "vy": command[1], "om": command[2]})


@app.post("/api/stop")
def api_stop():
    streamer.set_command((0.0, 0.0, 0.0))
    return jsonify({"ok": True})


@app.post("/api/shutdown")
def api_shutdown():
    payload = request.get_json(force=True, silent=True) or {}
    if payload.get("confirm") != "shutdown":
        return jsonify({"error": "Missing shutdown confirmation"}), 400

    try:
        project_player.stop()
        animation_player.stop()
        prop_broadcaster.reset_defaults()
        streamer.set_command((0.0, 0.0, 0.0))
        subprocess.Popen(shlex.split(SHUTDOWN_COMMAND))
    except Exception as exc:
        return jsonify({"error": f"Shutdown command failed: {exc}"}), 500

    return jsonify({"ok": True, "message": "Shutdown command sent"})


@app.get("/api/animations")
def api_animations():
    return jsonify({"animations": animation_player.list_files(), "status": animation_player.status()})


@app.post("/api/animations/upload")
def api_animation_upload():
    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"error": "Missing file field"}), 400

    try:
        filename = animation_player.save_upload(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"ok": True, "filename": filename, "animations": animation_player.list_files()})


@app.post("/api/animations/play")
def api_animation_play():
    payload = request.get_json(force=True, silent=True) or {}
    filename = str(payload.get("filename", ""))

    try:
        animation_player.play(filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409

    return jsonify({"ok": True, "status": animation_player.status()})


@app.post("/api/animations/stop")
def api_animation_stop():
    animation_player.stop()
    return jsonify({"ok": True, "status": animation_player.status()})


@app.get("/api/animations/status")
def api_animation_status():
    return jsonify(animation_player.status())


@app.get("/api/project-assets")
def api_project_assets():
    return jsonify(
        {
            "animations": animation_player.list_files(),
            "audio": list_audio_files(),
            "scripts": list_script_files(),
            "projects": list_project_files(),
            "projectStatus": project_player.status(),
            "propStatus": prop_broadcaster.status(),
            "limitSeconds": PROJECT_LIMIT_SECONDS,
            "animationFps": ANIMATION_FPS,
        }
    )


@app.post("/api/audio/upload")
def api_audio_upload():
    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"error": "Missing file field"}), 400

    try:
        filename = project_player.save_audio_upload(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"ok": True, "filename": filename, "audio": list_audio_files()})


@app.get("/api/projects")
def api_projects():
    return jsonify({"projects": list_project_files(), "status": project_player.status()})


@app.get("/api/projects/<path:filename>")
def api_project_load(filename: str):
    try:
        project = project_player.load_project(filename)
    except (ValueError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"project": project})


@app.post("/api/projects/save")
def api_project_save():
    payload = request.get_json(force=True, silent=True) or {}
    filename = str(payload.get("filename") or payload.get("title") or "show_project")
    project_payload = payload.get("project") if isinstance(payload.get("project"), dict) else payload

    try:
        result = project_player.save_project(filename, project_payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"ok": True, **result, "projects": list_project_files()})


@app.post("/api/projects/play")
def api_project_play():
    payload = request.get_json(force=True, silent=True) or {}
    filename = str(payload.get("filename") or "")
    try:
        project_player.play(filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"ok": True, "status": project_player.status()})


@app.post("/api/projects/pause")
def api_project_pause():
    project_player.pause()
    return jsonify({"ok": True, "status": project_player.status()})


@app.post("/api/projects/resume")
def api_project_resume():
    project_player.resume()
    return jsonify({"ok": True, "status": project_player.status()})


@app.post("/api/projects/stop")
def api_project_stop():
    project_player.stop()
    return jsonify({"ok": True, "status": project_player.status()})


@app.get("/api/projects/status")
def api_project_status():
    return jsonify(project_player.status())


if __name__ == "__main__":
    print(f"Serving kiwi controller on http://0.0.0.0:8000")
    print(f"Serial target: {SERIAL_PORT} @ {BAUD_RATE}")
    print(
        f"Animation target: {ANIMATION_SERIAL_PORT} @ {ANIMATION_BAUD_RATE}, "
        f"{ANIMATION_FPS:g} fps, {ANIMATION_START_DELAY:g}s start delay"
    )
    print(f"Prop UDP target: {PROP_UDP_HOST}:{PROP_UDP_PORT} @ {PROP_UDP_HZ:g} Hz")
    print(f"Project assets: {AUDIO_DIR}, {PROJECT_DIR}, {SCRIPT_DIR}")
    app.run(host="0.0.0.0", port=8000, threaded=True)
