import os
import random
import socket
import subprocess
import time
from pathlib import Path

from agno.tools import tool

MUSIC_DIR = os.path.join(os.getcwd(), "AiriMusicFolder")
VLC_RC_HOST = "localhost"
VLC_RC_PORT = 9191

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".flac",
    ".wav",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
}


_vlc_process: subprocess.Popen | None = None


def _ensure_music_dir() -> Path:
    path = Path(MUSIC_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_all_songs() -> list[Path]:
    music_dir = _ensure_music_dir()
    songs = sorted(
        [f for f in music_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    return songs


def _is_vlc_running() -> bool:
    global _vlc_process
    return _vlc_process is not None and _vlc_process.poll() is None


def _start_vlc() -> bool:
    """Starts VLC as a background daemon with RC interface for control."""
    global _vlc_process

    if _is_vlc_running():
        return True

    try:
        _vlc_process = subprocess.Popen(
            [
                "vlc",
                "--intf",
                "rc",
                "--rc-host",
                f"{VLC_RC_HOST}:{VLC_RC_PORT}",
                "--no-video",
                "--quiet",
                "--play-and-stop",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)
        return _is_vlc_running()
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _send_vlc_command(command: str) -> str:
    """Sends a command to VLC's RC interface and returns the response."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            s.connect((VLC_RC_HOST, VLC_RC_PORT))
            s.recv(1024)
            s.sendall((command + "\n").encode())
            time.sleep(0.2)
            try:
                response = s.recv(4096).decode().strip()
            except socket.timeout:
                response = ""
            return response
    except ConnectionRefusedError:
        return "error: VLC not responding"
    except Exception as e:
        return f"error: {e}"


def _stop_and_clear_vlc():
    """Stops playback and clears the VLC playlist."""
    _send_vlc_command("stop")
    _send_vlc_command("clear")


@tool
def list_songs(max_results: int = 100) -> str:
    """
    Lists all music files available in the AiriMusicFolder.

    Args:
        max_results: Maximum number of songs to list (default 100).

    Returns:
        A formatted list of available songs with their index and file name.
    """
    songs = _get_all_songs()

    if not songs:
        return (
            f"No music files found in '{MUSIC_DIR}'.\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    songs = songs[:max_results]
    lines = [f"🎵 Music Library — {len(songs)} song(s) in {MUSIC_DIR}:\n"]
    for i, song in enumerate(songs, start=1):
        size_mb = song.stat().st_size / (1024 * 1024)
        lines.append(f"  {i:>3}. {song.stem}  ({size_mb:.1f} MB)")

    return "\n".join(lines)


@tool
def play_song(song_name: str) -> str:
    """
    Plays a specific song from the AiriMusicFolder by name.
    A partial name match is supported (e.g., 'bohemian' will match 'Bohemian Rhapsody.mp3').

    Args:
        song_name: The name (or partial name) of the song to play.

    Returns:
        A status message confirming playback or describing an error.
    """
    songs = _get_all_songs()

    if not songs:
        return f"No music found in '{MUSIC_DIR}'."

    query = song_name.strip().lower()
    match = next(
        (s for s in songs if query in s.stem.lower()),
        None,
    )

    if not match:
        available = "\n".join(f"  • {s.stem}" for s in songs[:10])
        return (
            f"No song matching '{song_name}' found.\n"
            f"Available songs (first 10):\n{available}"
        )

    if not _start_vlc():
        return "Error: VLC is not installed or failed to start. Install VLC to use music playback."

    _stop_and_clear_vlc()
    _send_vlc_command(f"add {match}")
    _send_vlc_command("play")

    return f"▶ Now playing: {match.stem}"


@tool
def play_playlist() -> str:
    """
    Plays all songs in the AiriMusicFolder in alphabetical order.

    Returns:
        A status message confirming playback has started.
    """
    songs = _get_all_songs()

    if not songs:
        return f"No music found in '{MUSIC_DIR}'."

    if not _start_vlc():
        return "Error: VLC is not installed or failed to start."

    _stop_and_clear_vlc()

    for song in songs:
        _send_vlc_command(f"enqueue {song}")

    _send_vlc_command("random off")
    _send_vlc_command("loop on")
    _send_vlc_command("play")

    preview = "\n".join(f"  {i + 1}. {s.stem}" for i, s in enumerate(songs[:5]))
    if len(songs) > 5:
        preview += f"\n  ... and {len(songs) - 5} more"

    return f"▶ Playing full playlist — {len(songs)} song(s)\n{preview}"


@tool
def play_random() -> str:
    """
    Picks a random song from the AiriMusicFolder and plays it.
    Shuffles the full playlist so playback continues in random order.

    Returns:
        A status message showing which song is playing first.
    """
    songs = _get_all_songs()

    if not songs:
        return f"No music found in '{MUSIC_DIR}'."

    if not _start_vlc():
        return "Error: VLC is not installed or failed to start."

    shuffled = songs.copy()
    random.shuffle(shuffled)

    _stop_and_clear_vlc()

    for song in shuffled:
        _send_vlc_command(f"enqueue {song}")

    _send_vlc_command("random on")
    _send_vlc_command("loop on")
    _send_vlc_command("play")

    return f"🔀 Shuffle on — Now playing: {shuffled[0].stem}"


@tool
def stop_music() -> str:
    """
    Stops music playback and clears the current playlist.

    Returns:
        A status message confirming playback has stopped.
    """
    if not _is_vlc_running():
        return "⏹ No music is currently playing."

    _stop_and_clear_vlc()
    return "⏹ Music stopped."


@tool
def pause_music() -> str:
    """
    Pauses or resumes music playback (toggle).

    Returns:
        A status message confirming the action.
    """
    if not _is_vlc_running():
        return "No music is currently playing."

    _send_vlc_command("pause")
    return "⏸ Playback paused / resumed."


@tool
def next_song() -> str:
    """
    Skips to the next song in the playlist.

    Returns:
        A status message confirming the skip.
    """
    if not _is_vlc_running():
        return "No music is currently playing."

    _send_vlc_command("next")
    return "⏭ Skipped to next song."


@tool
def previous_song() -> str:
    """
    Goes back to the previous song in the playlist.

    Returns:
        A status message confirming the action.
    """
    if not _is_vlc_running():
        return "No music is currently playing."

    _send_vlc_command("prev")
    return "⏮ Went back to previous song."


@tool
def set_volume(level: int) -> str:
    """
    Sets the playback volume.

    Args:
        level: Volume level from 0 (mute) to 100 (max).

    Returns:
        A status message confirming the volume change.
    """
    if not _is_vlc_running():
        return "No music is currently playing."

    level = max(0, min(100, level))

    vlc_volume = int(level * 2.56)
    _send_vlc_command(f"volume {vlc_volume}")
    return f"🔊 Volume set to {level}%."
