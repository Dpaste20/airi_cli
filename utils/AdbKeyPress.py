import asyncio
from typing import Optional

from agno.tools import tool

KEYCODES: dict[str, int] = {
    "enter": 66,
    "search": 84,
    "go": 66,
    "done": 66,
    "back": 4,
    "home": 3,
    "recents": 187,
    "tab": 61,
    "escape": 111,
    "dpad_up": 19,
    "dpad_down": 20,
    "dpad_left": 21,
    "dpad_right": 22,
    "dpad_center": 23,
    "space": 62,
    "clear": 28,
    "delete": 67,
    "forward_delete": 112,
    "select_all": 232,
    "copy": 278,
    "paste": 279,
    "cut": 277,
    "play_pause": 85,
    "stop": 86,
    "next": 87,
    "previous": 88,
    "volume_up": 24,
    "volume_down": 25,
    "mute": 164,
    "power": 26,
    "wakeup": 224,
    "sleep": 223,
    "menu": 82,
    "camera": 27,
    "shift": 59,
    "ctrl": 113,
    "alt": 57,
}


_REPEAT_DELAY = 0.15


async def _adb(
    args: list[str],
    device_id: Optional[str] = None,
    timeout: int = 15,
) -> str:
    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]
    cmd += args

    print(f"[AdbKeyPress] Running: {' '.join(cmd)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return "Error: adb command timed out."

        out = stdout.decode().strip()
        err = stderr.decode().strip()

        if process.returncode == 0:
            return out if out else "Key event sent successfully."
        else:
            return f"Error (exit {process.returncode}): {err or out}"

    except FileNotFoundError:
        return (
            "Error: `adb` not found. "
            "Install Android SDK platform-tools and ensure adb is on PATH."
        )
    except Exception as exc:
        return f"Execution exception: {exc}"


@tool
async def adb_key_press(
    key: str,
    device_id: Optional[str] = None,
    repeat: int = 1,
) -> str:
    """
    Press a keyboard or hardware key on a connected Android device using ADB.

    Use this immediately after agent_device fill to submit a search query,
    confirm a form, dismiss the keyboard, or trigger any IME action — since
    agent-device has no native key-press command.

    Common keys:
        "enter"      — confirm / submit (IME action, most text fields)
        "search"     — trigger in-app search (YouTube, Chrome, etc.)
        "tab"        — move focus to next field
        "back"       — hardware Back button
        "home"       — go to home screen
        "space"      — insert a space character
        "clear"      — clear the focused input field
        "delete"     — backspace one character
        "select_all" / "copy" / "paste" / "cut" — clipboard shortcuts
        "dpad_up/down/left/right/center" — D-pad (Android TV, game controllers)
        "volume_up" / "volume_down" / "mute" — media volume
        "play_pause" / "next" / "previous" / "stop" — media playback

    Full key list:
        enter, search, go, done, back, home, recents, tab, escape,
        dpad_up, dpad_down, dpad_left, dpad_right, dpad_center,
        space, clear, delete, forward_delete, select_all, copy, paste, cut,
        play_pause, stop, next, previous, volume_up, volume_down, mute,
        power, wakeup, sleep, menu, camera, shift, ctrl, alt

    You can also pass a raw integer keycode directly (e.g. key="66").

    Args:
        key:       Key name (see above) or raw ADB keycode integer as a string.
        device_id: Specific ADB device serial (e.g. "emulator-5554" or
                   "R3CN90XXXXX"). Leave blank to target the only connected device.
        repeat:    Number of times to press the key (default 1).
                   A short delay is inserted between presses to ensure each
                   event registers on the device.

    Returns:
        "Key event sent successfully." on success, or error details.

    Example flow — YouTube search on Android:
        1. agent_device("open YouTube --platform android")
        2. agent_device("find 'Search' click")
        3. agent_device("fill @e3 'F-22 Raptor'")
        4. adb_key_press(key="search")          ← this tool

    Example flow — delete last 5 characters:
        adb_key_press(key="delete", repeat=5)
    """

    key_lower = key.lower().strip()

    if key_lower in KEYCODES:
        keycode = KEYCODES[key_lower]
    else:
        try:
            keycode = int(key)
        except ValueError:
            suggestions = [k for k in KEYCODES if key_lower in k]
            hint = f" Did you mean: {suggestions}?" if suggestions else ""
            return (
                f"Error: unknown key '{key}'.{hint} "
                f"Supported keys: {sorted(KEYCODES.keys())}"
            )

    results = []
    for i in range(max(1, repeat)):
        result = await _adb(
            ["shell", "input", "keyevent", str(keycode)],
            device_id=device_id,
        )
        results.append(result)

        if "Error" in result:
            break

        if i < repeat - 1:
            await asyncio.sleep(_REPEAT_DELAY)

    unique = list(dict.fromkeys(results))
    if len(unique) == 1:
        return unique[0]
    return "\n".join(results)
