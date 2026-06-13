from __future__ import annotations

import argparse
import ctypes
import json
import logging
import mimetypes
import os
import socket
import subprocess
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MODELS_DIR = APP_DIR / "models"
RUNTIME_DIR = APP_DIR / ".runtime"
JOINT_OFFSET_FILE = RUNTIME_DIR / "joint_offset.json"
DATASETS_DIR = Path(
    os.getenv(
        "MINIFAB_DATASETS_DIR",
        str(Path.home() / ".cache" / "huggingface" / "lerobot" / "local"),
    )
)
ENV_ROOT = Path(
    os.getenv(
        "MINIFAB_ENV_ROOT",
        str(Path.home() / "miniconda3" / "envs" / "lerobot-win"),
    )
)
RECORD_EXE = ENV_ROOT / "Scripts" / "lerobot-record.exe"
EDIT_DATASET_EXE = ENV_ROOT / "Scripts" / "lerobot-edit-dataset.exe"
RERUN_EXE = ENV_ROOT / "Scripts" / "rerun.exe"
PYTHON_EXE = ENV_ROOT / "python.exe"
WINDOWS_VK = {"left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "escape": 0x1B}
JOINT_OFFSET_MIN = -5.0
JOINT_OFFSET_MAX = 5.0
DEFAULT_JOINT_OFFSET = 0.13


def normalize_repo_id(value: str) -> str:
    value = value.strip().strip("'\"")
    if not value:
        raise ValueError("Dataset name cannot be empty.")
    if value.startswith("local/"):
        return value
    return f"local/{value}"


def build_camera_config(
    front_id: int,
    side_id: int,
    width: int,
    height: int,
    fps: int,
    fourcc: str,
    warmup_s: int,
) -> str:
    fourcc = fourcc.strip().upper()
    return (
        "{"
        f'front: {{type: opencv, index_or_path: {front_id}, width: {width}, height: {height}, fps: {fps}, fourcc: "{fourcc}"}}, '
        f'side: {{type: opencv, index_or_path: {side_id}, width: {width}, height: {height}, fps: {fps}, fourcc: "{fourcc}", warmup_s: {warmup_s}}}'
        "}"
    )


def bool_flag(value: bool) -> str:
    return "true" if value else "false"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def ensure_runtime_state() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not JOINT_OFFSET_FILE.exists():
        JOINT_OFFSET_FILE.write_text(json.dumps({"joint_1_offset": DEFAULT_JOINT_OFFSET}), encoding="utf-8")


def get_joint_offset_state() -> dict[str, float]:
    ensure_runtime_state()
    payload = read_json(JOINT_OFFSET_FILE, default={}) or {}
    try:
        value = float(payload.get("joint_1_offset", DEFAULT_JOINT_OFFSET))
    except (TypeError, ValueError):
        value = DEFAULT_JOINT_OFFSET
    value = max(JOINT_OFFSET_MIN, min(JOINT_OFFSET_MAX, value))
    return {"joint_1_offset": value}


def write_joint_offset_state(value: float) -> dict[str, float]:
    ensure_runtime_state()
    value = max(JOINT_OFFSET_MIN, min(JOINT_OFFSET_MAX, float(value)))
    payload = {"joint_1_offset": value}
    tmp_path = JOINT_OFFSET_FILE.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(JOINT_OFFSET_FILE)
    return payload


def build_runtime_env(display_data: bool) -> dict[str, str]:
    ensure_runtime_state()
    env = {"MINIFAB_JOINT_OFFSET_FILE": str(JOINT_OFFSET_FILE)}
    if display_data:
        RERUN_MANAGER.ensure_running()
        env["LEROBOT_RERUN_CONNECT_URL"] = RERUN_MANAGER.connect_url
    return env


def emit_virtual_key(key_name: str) -> None:
    if key_name not in WINDOWS_VK:
        raise ValueError(f"Unsupported key: {key_name}")
    vk = WINDOWS_VK[key_name]
    user32 = ctypes.windll.user32
    scan = user32.MapVirtualKeyW(vk, 0)
    user32.keybd_event(vk, scan, 0, 0)
    time.sleep(0.03)
    user32.keybd_event(vk, scan, 0x0002, 0)


@dataclass
class RunningProcess:
    mode: str
    command: list[str]
    process: subprocess.Popen[str]
    started_at: float
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=800))
    active_model: str | None = None
    preloaded_models: list[str] = field(default_factory=list)
    returncode: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "running": self.process.poll() is None,
            "pid": self.process.pid,
            "started_at": self.started_at,
            "returncode": self.returncode if self.returncode is not None else self.process.poll(),
            "command": subprocess.list2cmdline(self.command),
            "active_model": self.active_model,
            "preloaded_models": self.preloaded_models,
            "logs": list(self.logs),
        }


class ProcessManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: RunningProcess | None = None

    def _reader(self, running: RunningProcess) -> None:
        assert running.process.stdout is not None
        for line in running.process.stdout:
            running.logs.append(line.rstrip())
        running.returncode = running.process.wait()
        running.logs.append(f"[process exited with code {running.returncode}]")

    def start(
        self,
        *,
        mode: str,
        command: list[str],
        active_model: str | None = None,
        preloaded_models: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> RunningProcess:
        with self._lock:
            if self._running and self._running.process.poll() is None:
                raise RuntimeError("Another LeRobot session is already running.")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if extra_env:
                env.update(extra_env)
            process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            running = RunningProcess(
                mode=mode,
                command=command,
                process=process,
                started_at=time.time(),
                active_model=active_model,
                preloaded_models=preloaded_models or [],
            )
            running.logs.append(f"[started] {subprocess.list2cmdline(command)}")
            threading.Thread(target=self._reader, args=(running,), daemon=True).start()
            self._running = running
            return running

    def current(self) -> RunningProcess | None:
        with self._lock:
            if self._running and self._running.process.poll() is not None:
                self._running.returncode = self._running.process.poll()
            return self._running

    def clear_if_finished(self) -> None:
        return

    def stop(self) -> dict[str, Any]:
        running = self.current()
        if running is None or running.process.poll() is not None:
            raise RuntimeError("No active LeRobot session.")
        emit_virtual_key("escape")
        running.logs.append("[control] sent Escape")
        return running.to_dict()

    def terminate(self) -> dict[str, Any]:
        running = self.current()
        if running is None or running.process.poll() is not None:
            raise RuntimeError("No active LeRobot session.")
        running.process.terminate()
        running.logs.append("[control] terminate requested")
        return running.to_dict()


@dataclass
class BackgroundJob:
    job_id: str
    label: str
    command: list[str]
    started_at: float
    status: str = "running"
    returncode: int | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=400))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "label": self.label,
            "status": self.status,
            "started_at": self.started_at,
            "returncode": self.returncode,
            "command": subprocess.list2cmdline(self.command),
            "logs": list(self.logs),
        }


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, BackgroundJob] = {}
        self._counter = 0

    def start(self, label: str, command: list[str]) -> BackgroundJob:
        with self._lock:
            self._counter += 1
            job = BackgroundJob(
                job_id=f"job-{self._counter}",
                label=label,
                command=command,
                started_at=time.time(),
            )
            self._jobs[job.job_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: BackgroundJob) -> None:
        job.logs.append(f"[started] {subprocess.list2cmdline(job.command)}")
        process = subprocess.Popen(
            job.command,
            cwd=str(APP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            job.logs.append(line.rstrip())
        job.returncode = process.wait()
        job.status = "completed" if job.returncode == 0 else "failed"
        job.logs.append(f"[job exited with code {job.returncode}]")

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in sorted(self._jobs.values(), key=lambda item: item.started_at, reverse=True)]


class CameraFeed:
    def __init__(self, camera_id: int, width: int, height: int, fps: int, fourcc: str) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.fourcc = fourcc
        self._lock = threading.Lock()
        self._frame: bytes | None = None
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.fourcc and len(self.fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    with self._lock:
                        self._frame = encoded.tobytes()
                time.sleep(1 / self.fps)
        finally:
            cap.release()

    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame


class CameraRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._feeds: dict[tuple[int, int, int, int, str], CameraFeed] = {}

    def get(self, camera_id: int, width: int, height: int, fps: int, fourcc: str) -> CameraFeed:
        key = (camera_id, width, height, fps, fourcc.upper())
        with self._lock:
            feed = self._feeds.get(key)
            if feed is None:
                feed = CameraFeed(*key)
                self._feeds[key] = feed
            return feed


class RerunManager:
    def __init__(self, grpc_port: int = 9876, web_port: int = 9090) -> None:
        self.grpc_port = grpc_port
        self.web_port = web_port
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    @property
    def viewer_url(self) -> str:
        return f"http://127.0.0.1:{self.web_port}/"

    @property
    def connect_url(self) -> str:
        return f"rerun+http://127.0.0.1:{self.grpc_port}/proxy"

    def _port_open(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def ensure_running(self) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                return
            if self._port_open(self.grpc_port) and self._port_open(self.web_port):
                return
            self._process = subprocess.Popen(
                [
                    str(RERUN_EXE),
                    "--bind",
                    "127.0.0.1",
                    "--serve-web",
                    "--port",
                    str(self.grpc_port),
                    "--web-viewer-port",
                    str(self.web_port),
                    "--hide-welcome-screen",
                ],
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            deadline = time.time() + 5
            while time.time() < deadline:
                if self._port_open(self.grpc_port) and self._port_open(self.web_port):
                    return
                if self._process.poll() is not None:
                    break
                time.sleep(0.15)
            raise RuntimeError("Failed to start the embedded Rerun web viewer.")

    def shutdown(self) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()


def list_serial_ports() -> list[dict[str, str]]:
    script = (
        "from serial.tools import list_ports\n"
        "import json\n"
        "print(json.dumps([{'device': p.device, 'description': p.description} for p in list_ports.comports()]))\n"
    )
    result = subprocess.run(
        [str(PYTHON_EXE), "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return json.loads(result.stdout)


def probe_cameras(max_index: int = 6) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if ok:
            cameras.append({"index": index, "label": f"Camera {index}"})
    return cameras


def dataset_summary(dataset_dir: Path) -> dict[str, Any]:
    info = read_json(dataset_dir / "meta" / "info.json", default={}) or {}
    feature_keys = list((info.get("features") or {}).keys())
    image_features = [key for key in feature_keys if key.startswith("observation.images.")]
    return {
        "name": dataset_dir.name,
        "repo_id": f"local/{dataset_dir.name}",
        "path": str(dataset_dir),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "data_files_size_in_mb": info.get("data_files_size_in_mb"),
        "video_files_size_in_mb": info.get("video_files_size_in_mb"),
        "feature_keys": feature_keys,
        "image_features": image_features,
        "info": info,
    }


def list_datasets() -> list[dict[str, Any]]:
    if not DATASETS_DIR.exists():
        return []
    items = []
    for dataset_dir in sorted(DATASETS_DIR.iterdir()):
        if dataset_dir.is_dir() and (dataset_dir / "meta" / "info.json").exists():
            items.append(dataset_summary(dataset_dir))
    return items


def model_summary(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path, default={}) or {}
    pretrained_dir = config_path.parent
    checkpoint_dir = pretrained_dir.parent
    task_dir = checkpoint_dir.parent.parent
    return {
        "label": f"{task_dir.name} / {checkpoint_dir.name}",
        "task": task_dir.name,
        "checkpoint": checkpoint_dir.name,
        "path": str(pretrained_dir),
        "config_path": str(config_path),
        "type": config.get("type"),
        "n_action_steps": config.get("n_action_steps"),
        "input_features": list((config.get("input_features") or {}).keys()),
    }


def list_models() -> list[dict[str, Any]]:
    if not MODELS_DIR.exists():
        return []
    items = [model_summary(path) for path in MODELS_DIR.glob("*/checkpoints/*/pretrained_model/config.json")]
    return sorted(items, key=lambda item: (item["task"], item["checkpoint"]))


def build_record_command(payload: dict[str, Any]) -> list[str]:
    dataset_repo_id = normalize_repo_id(payload["dataset_name"])
    camera_config = build_camera_config(
        int(payload["front_camera_id"]),
        int(payload["side_camera_id"]),
        int(payload["camera_width"]),
        int(payload["camera_height"]),
        int(payload["dataset_fps"]),
        str(payload["camera_fourcc"]),
        int(payload["side_warmup_s"]),
    )
    single_task = payload.get("single_task", "")
    return [
        str(RECORD_EXE),
        "--robot.type=serial_follower",
        f"--robot.port={payload['follower_port']}",
        "--teleop.type=serial_leader",
        f"--teleop.port={payload['leader_port']}",
        f"--robot.cameras={camera_config}",
        f"--dataset.num_episodes={int(payload['num_episodes'])}",
        f"--dataset.fps={int(payload['dataset_fps'])}",
        "--dataset.push_to_hub=false",
        f"--dataset.single_task={single_task}",
        f"--display_data={bool_flag(bool(payload.get('display_data', False)))}",
        "--play_sounds=false",
        f"--dataset.episode_time_s={int(payload['episode_time_s'])}",
        f"--dataset.repo_id={dataset_repo_id}",
    ]


def build_inference_command(payload: dict[str, Any]) -> tuple[list[str], str, list[str]]:
    active_model = str(payload["active_model"])
    if not active_model or not Path(active_model).exists():
        raise ValueError("Choose a valid active model checkpoint.")
    preload_models = [str(path) for path in payload.get("preload_models", []) if str(path).strip()]
    camera_config = build_camera_config(
        int(payload["front_camera_id"]),
        int(payload["side_camera_id"]),
        int(payload["camera_width"]),
        int(payload["camera_height"]),
        int(payload["fps"]),
        str(payload["camera_fourcc"]),
        int(payload["side_warmup_s"]),
    )
    command = [
        str(RECORD_EXE),
        "--robot.type=serial_follower",
        f"--robot.port={payload['follower_port']}",
        f"--robot.cameras={camera_config}",
        f"--display_data={bool_flag(bool(payload.get('display_data', False)))}",
        f"--dataset.num_episodes={int(payload['num_episodes'])}",
        "--play_sounds=false",
        f"--dataset.episode_time_s={int(payload['episode_time_s'])}",
        f"--dataset.fps={int(payload['fps'])}",
        f"--policy.path={active_model}",
        f"--policy.n_action_steps={int(payload['n_action_steps'])}",
    ]
    if preload_models:
        command.append(f"--preloaded_policy_paths={';'.join(preload_models)}")
    single_task = str(payload.get("single_task", "")).strip()
    if single_task:
        command.append(f"--dataset.single_task={single_task}")
    return command, active_model, preload_models


def load_static(relative_path: str) -> bytes:
    path = (STATIC_DIR / relative_path).resolve()
    if STATIC_DIR not in path.parents and path != STATIC_DIR:
        raise FileNotFoundError(relative_path)
    return path.read_bytes()


PROCESS_MANAGER = ProcessManager()
JOB_MANAGER = JobManager()
CAMERA_REGISTRY = CameraRegistry()
RERUN_MANAGER = RerunManager()


class SkaioHandler(BaseHTTPRequestHandler):
    server_version = "SkaioHTTP/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        logging.info("%s - %s", self.client_address[0], format % args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, relative_path: str) -> None:
        try:
            body = load_static(relative_path)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime, _ = mimetypes.guess_type(relative_path)
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        PROCESS_MANAGER.clear_if_finished()
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_static("index.html")
            return
        if parsed.path in {"/app.js", "/styles.css"}:
            self._serve_static(parsed.path.lstrip("/"))
            return
        if parsed.path == "/api/state":
            current = PROCESS_MANAGER.current()
            self._send_json(
                {
                    "process": current.to_dict() if current else None,
                    "jobs": JOB_MANAGER.list(),
                    "joint_offset": get_joint_offset_state(),
                    "rerun": {"viewer_url": RERUN_MANAGER.viewer_url, "connect_url": RERUN_MANAGER.connect_url},
                }
            )
            return
        if parsed.path == "/api/datasets":
            self._send_json({"datasets": list_datasets()})
            return
        if parsed.path == "/api/models":
            self._send_json({"models": list_models()})
            return
        if parsed.path == "/api/ports":
            self._send_json({"ports": list_serial_ports()})
            return
        if parsed.path == "/api/cameras/probe":
            query = parse_qs(parsed.query)
            max_index = int(query.get("max_index", ["6"])[0])
            self._send_json({"cameras": probe_cameras(max_index=max_index)})
            return
        if parsed.path == "/api/camera/stream":
            query = parse_qs(parsed.query)
            camera_id = int(query.get("camera_id", ["0"])[0])
            width = int(query.get("width", ["640"])[0])
            height = int(query.get("height", ["480"])[0])
            fps = int(query.get("fps", ["15"])[0])
            fourcc = query.get("fourcc", ["MJPG"])[0]
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            feed = CAMERA_REGISTRY.get(camera_id, width, height, fps, fourcc)
            try:
                while True:
                    frame = feed.frame()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        PROCESS_MANAGER.clear_if_finished()
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/datasets/merge":
                payload = self._read_json()
                repo_id = normalize_repo_id(payload["target_repo_id"])
                source_repo_ids = [normalize_repo_id(item) for item in payload.get("source_repo_ids", [])]
                if len(source_repo_ids) < 2:
                    raise ValueError("Select at least two datasets to merge.")
                repo_list = "[" + ",".join(f"'{repo_id}'" for repo_id in source_repo_ids) + "]"
                command = [
                    str(EDIT_DATASET_EXE),
                    "--repo_id",
                    repo_id,
                    "--operation.type",
                    "merge",
                    "--operation.repo_ids",
                    repo_list,
                    "--push_to_hub",
                    "false",
                ]
                job = JOB_MANAGER.start(f"Merge {repo_id}", command)
                self._send_json({"job": job.to_dict()}, status=202)
                return
            if parsed.path == "/api/record/start":
                payload = self._read_json()
                extra_env = build_runtime_env(bool(payload.get("display_data", False)))
                running = PROCESS_MANAGER.start(
                    mode="record",
                    command=build_record_command(payload),
                    extra_env=extra_env,
                )
                self._send_json({"process": running.to_dict()}, status=202)
                return
            if parsed.path == "/api/inference/start":
                payload = self._read_json()
                command, active_model, preload_models = build_inference_command(payload)
                extra_env = build_runtime_env(bool(payload.get("display_data", False)))
                running = PROCESS_MANAGER.start(
                    mode="inference",
                    command=command,
                    active_model=active_model,
                    preloaded_models=preload_models,
                    extra_env=extra_env,
                )
                self._send_json({"process": running.to_dict()}, status=202)
                return
            if parsed.path == "/api/joint-offset":
                payload = self._read_json()
                state = write_joint_offset_state(payload.get("joint_1_offset", 0.0))
                current = PROCESS_MANAGER.current()
                if current:
                    current.logs.append(f"[control] joint_1 send offset {state['joint_1_offset']:+.2f}")
                self._send_json({"joint_offset": state})
                return
            if parsed.path == "/api/process/stop":
                self._send_json({"process": PROCESS_MANAGER.stop()})
                return
            if parsed.path == "/api/process/terminate":
                self._send_json({"process": PROCESS_MANAGER.terminate()})
                return
            if parsed.path == "/api/control/key":
                payload = self._read_json()
                key_name = str(payload["key"]).lower()
                emit_virtual_key(key_name)
                current = PROCESS_MANAGER.current()
                if current:
                    current.logs.append(f"[control] sent {key_name}")
                self._send_json({"ok": True, "key": key_name})
                return
        except subprocess.CalledProcessError as exc:
            self._send_error(exc.stderr or exc.stdout or str(exc), status=500)
            return
        except (ValueError, RuntimeError, KeyError) as exc:
            self._send_error(str(exc), status=400)
            return
        except Exception as exc:
            self._send_error(str(exc), status=500)
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def ensure_prerequisites() -> None:
    missing = [path for path in (RECORD_EXE, EDIT_DATASET_EXE, RERUN_EXE, PYTHON_EXE, STATIC_DIR / "index.html") if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing prerequisite files: {names}")
    ensure_runtime_state()


def run_server(host: str, port: int, open_browser: bool) -> None:
    ensure_prerequisites()
    server = ThreadingHTTPServer((host, port), SkaioHandler)
    url = f"http://{host}:{port}/"
    print(f"Skaio interface listening on {url}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down Skaio interface.")
    finally:
        RERUN_MANAGER.shutdown()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Skaio local interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_server(args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
