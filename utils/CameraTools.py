import os
import signal
import subprocess
import time
from pathlib import Path

from agno.tools import tool

CAPTURES_DIR = os.path.expanduser("~/Pictures/Airi")
_recording_process: subprocess.Popen | None = None
_recording_path: str = ""


def _ensure_captures_dir() -> Path:
    path = Path(CAPTURES_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _detect_camera_device() -> str | None:
    """Return the first available /dev/video* device, or None."""
    for i in range(10):
        dev = f"/dev/video{i}"
        if os.path.exists(dev):
            return dev
    return None


@tool
def take_picture(
    filename: str = "",
    camera_index: int = 0,
    delay_seconds: int = 0,
) -> str:
    """
    Captures a single photo from the webcam and saves it as a JPEG image.

    Args:
        filename       : Optional custom filename (without extension).
                         Defaults to a timestamped name like 'photo_20250305_142300'.
        camera_index   : Index of the camera device to use (default 0 → /dev/video0).
        delay_seconds  : Countdown delay before capturing (0 = immediate).

    Returns:
        Path to the saved image, or an error message.
    """
    save_dir = _ensure_captures_dir()
    name = filename.strip() if filename.strip() else f"photo_{_timestamp()}"
    output_path = str(save_dir / f"{name}.jpg")

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    if shutil.which("fswebcam"):
        cmd = [
            "fswebcam",
            "-d",
            f"/dev/video{camera_index}",
            "--no-banner",
            "-r",
            "1280x720",
            output_path,
        ]
    elif shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "v4l2",
            "-i",
            f"/dev/video{camera_index}",
            "-vframes",
            "1",
            "-q:v",
            "2",
            output_path,
        ]
    else:
        return "Error: Neither 'fswebcam' nor 'ffmpeg' is installed. Install one to capture photos."

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) // 1024
            return f"✓ Photo saved: {output_path} ({size_kb} KB)"
        else:
            return (
                f"Error capturing photo (exit {result.returncode}):\n"
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
    except subprocess.TimeoutExpired:
        return "Error: Camera capture timed out after 30 seconds."
    except Exception as e:
        return f"Camera exception: {e}"


@tool
def start_recording(
    filename: str = "",
    camera_index: int = 0,
    resolution: str = "1280x720",
    fps: int = 30,
    with_audio: bool = True,
) -> str:
    """
    Starts a video recording from the webcam in the background.
    Call stop_recording() to end and save the file.

    Args:
        filename      : Optional custom filename (without extension).
                        Defaults to 'video_<timestamp>'.
        camera_index  : Camera device index (default 0 → /dev/video0).
        resolution    : Recording resolution, e.g. '1280x720' or '1920x1080'.
        fps           : Frames per second (default 30).
        with_audio    : Whether to capture microphone audio alongside video (default True).

    Returns:
        Confirmation message with the output path, or an error message.
    """
    global _recording_process, _recording_path

    if _recording_process and _recording_process.poll() is None:
        return f"⚠ A recording is already in progress → {_recording_path}. Call stop_recording() first."

    if not shutil.which("ffmpeg"):
        return "Error: 'ffmpeg' is not installed. Install it to record video."

    save_dir = _ensure_captures_dir()
    name = filename.strip() if filename.strip() else f"video_{_timestamp()}"
    output_path = str(save_dir / f"{name}.mp4")
    _recording_path = output_path

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "v4l2",
        "-framerate",
        str(fps),
        "-video_size",
        resolution,
        "-i",
        f"/dev/video{camera_index}",
    ]

    if with_audio and shutil.which("ffmpeg"):
        cmd += ["-f", "pulse", "-i", "default"]

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    try:
        _recording_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        if _recording_process.poll() is not None:
            _recording_process = None
            return "Error: Recording process failed to start. Check camera device and permissions."

        return (
            f"🔴 Recording started → {output_path}\n"
            f"Resolution: {resolution} @ {fps} fps | Audio: {'on' if with_audio else 'off'}\n"
            "Call stop_recording() to finish."
        )
    except Exception as e:
        _recording_process = None
        return f"Recording exception: {e}"


@tool
def stop_recording() -> str:
    """
    Stops the currently active video recording and finalises the output file.

    Returns:
        Path to the saved video and its size, or an error message if no recording is active.
    """
    global _recording_process, _recording_path

    if not _recording_process or _recording_process.poll() is not None:
        _recording_process = None
        return "⚠ No active recording to stop."

    try:
        _recording_process.send_signal(signal.SIGINT)
        _recording_process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _recording_process.kill()
        _recording_process.wait()
    except Exception as e:
        return f"Error stopping recording: {e}"
    finally:
        _recording_process = None

    if os.path.exists(_recording_path):
        size_mb = os.path.getsize(_recording_path) / (1024 * 1024)
        saved_path = _recording_path
        _recording_path = ""
        return f"✓ Recording saved: {saved_path} ({size_mb:.1f} MB)"
    else:
        return "Recording stopped but output file was not found."


@tool
def get_recording_status() -> str:
    """
    Returns whether a video recording is currently active, and the target output file.

    Returns:
        Status string indicating active/idle state.
    """
    if _recording_process and _recording_process.poll() is None:
        return f"🔴 Recording is ACTIVE → {_recording_path}"
    return "⏹ No recording in progress."


@tool
def list_captures(media_type: str = "all") -> str:
    """
    Lists all photos and/or videos saved by CameraTools.

    Args:
        media_type : Filter results — 'photos', 'videos', or 'all' (default).

    Returns:
        A formatted list of captured files with sizes and timestamps, or a message if none exist.
    """
    save_dir = _ensure_captures_dir()

    extensions = {
        "photos": {".jpg", ".jpeg", ".png"},
        "videos": {".mp4", ".mkv", ".avi"},
        "all": {".jpg", ".jpeg", ".png", ".mp4", ".mkv", ".avi"},
    }.get(media_type.lower(), {".jpg", ".jpeg", ".png", ".mp4", ".mkv", ".avi"})

    files = sorted(
        [f for f in save_dir.iterdir() if f.suffix.lower() in extensions],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not files:
        return f"No {media_type} captures found in {save_dir}."

    lines = [f"Captures in {save_dir} ({len(files)} file(s)):\n"]
    for f in files:
        stat = f.stat()
        size = stat.st_size / 1024
        unit = "KB"
        if size > 1024:
            size /= 1024
            unit = "MB"
        modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        icon = "🖼" if f.suffix.lower() in {".jpg", ".jpeg", ".png"} else "🎬"
        lines.append(f"  {icon} {f.name}  —  {size:.1f} {unit}  —  {modified}")

    return "\n".join(lines)


@tool
def delete_capture(filename: str) -> str:
    """
    Deletes a specific photo or video from the captures directory.

    Args:
        filename : Name of the file to delete (with or without extension).
                   Example: 'photo_20250305_142300' or 'video_20250305_150000.mp4'

    Returns:
        Confirmation or error message.
    """
    save_dir = _ensure_captures_dir()

    target = save_dir / filename
    if not target.exists():
        matches = list(save_dir.glob(f"{filename}.*"))
        if not matches:
            return f"File not found: '{filename}' in {save_dir}"
        target = matches[0]

    try:
        target.unlink()
        return f"Deleted: {target.name}"
    except Exception as e:
        return f"Error deleting file: {e}"


@tool
def take_timelapse(
    total_shots: int = 10,
    interval_seconds: int = 5,
    camera_index: int = 0,
    filename_prefix: str = "",
) -> str:
    """
    Captures a series of photos at a fixed interval for timelapse creation.

    Args:
        total_shots       : Number of photos to capture (default 10).
        interval_seconds  : Seconds between each shot (default 5).
        camera_index      : Camera device index (default 0).
        filename_prefix   : Optional prefix for output filenames.

    Returns:
        Summary of captured frames and their save location.
    """
    save_dir = _ensure_captures_dir()
    prefix = filename_prefix.strip() or f"timelapse_{_timestamp()}"
    session_dir = save_dir / prefix
    session_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("fswebcam") and not shutil.which("ffmpeg"):
        return "Error: Neither 'fswebcam' nor 'ffmpeg' is installed."

    captured = 0
    failed = 0

    for i in range(total_shots):
        shot_path = str(session_dir / f"frame_{i + 1:04d}.jpg")

        if shutil.which("fswebcam"):
            cmd = [
                "fswebcam",
                "-d",
                f"/dev/video{camera_index}",
                "--no-banner",
                "-r",
                "1280x720",
                shot_path,
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "v4l2",
                "-i",
                f"/dev/video{camera_index}",
                "-vframes",
                "1",
                "-q:v",
                "2",
                shot_path,
            ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=20)
            if result.returncode == 0 and os.path.exists(shot_path):
                captured += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        if i < total_shots - 1:
            time.sleep(interval_seconds)

    total_duration = total_shots * interval_seconds
    return (
        f"✓ Timelapse complete: {captured}/{total_shots} frames captured"
        + (f" ({failed} failed)" if failed else "")
        + f"\nSaved to: {session_dir}"
        + f"\nTotal span: ~{total_duration}s  |  Interval: {interval_seconds}s per frame"
    )


import shutil  # noqa: E402
