import asyncio
import os

from agno.tools import tool

OBSERVATION_COMMANDS = (
    "snapshot",
    "diff",
    "get ",
    "is ",
    "find ",
    "devices",
    "session",
    "screenshot",
    "batch",
    "logs",
    "appstate",
    "apps",
    "clipboard read",
    "perf",
    "metrics",
    "network",
)


PLATFORM_COMMANDS = ("open", "boot", "push", "apps", "screenshot")


_TIMEOUT_BOOT = 180
_TIMEOUT_OBSERVATION = 180
_TIMEOUT_ACTION = 45


_detected_platform: str | None = None
_platform_detected: bool = False


def _needs_output(args: str) -> bool:
    stripped = args.strip().lower()
    return any(stripped.startswith(cmd) for cmd in OBSERVATION_COMMANDS)


def _get_timeout(args: str, wait_for_output: bool) -> int:
    if args.strip().lower().startswith("boot"):
        return _TIMEOUT_BOOT
    return _TIMEOUT_OBSERVATION if wait_for_output else _TIMEOUT_ACTION


def _needs_platform(args: str) -> bool:
    """Return True if the command is one that requires --platform."""
    stripped = args.strip().lower()
    return any(stripped.startswith(cmd) for cmd in PLATFORM_COMMANDS)


def _has_platform_flag(args: str) -> bool:
    """Return True if --platform is already present in args."""
    return "--platform" in args.lower()


async def _detect_platform() -> str | None:
    """
    Run 'agent-device devices' once and infer the platform from the output.
    Returns 'ios', 'android', or None if detection fails.
    """
    global _detected_platform, _platform_detected

    if _platform_detected:
        return _detected_platform

    print("Auto-detecting device platform...")
    try:
        process = await asyncio.create_subprocess_shell(
            "agent-device devices",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=20)
        output = stdout.decode().lower()

        if (
            "ios" in output
            or "iphone" in output
            or "ipad" in output
            or "simulator" in output
        ):
            _detected_platform = "ios"
        elif "android" in output or "emulator" in output or "avd" in output:
            _detected_platform = "android"
        else:
            _detected_platform = None

        _platform_detected = True
        if _detected_platform:
            print(f"Detected platform: {_detected_platform}")
        else:
            print("Platform detection inconclusive — no --platform will be injected.")
    except Exception as e:
        print(f"Platform detection failed: {e}")
        _platform_detected = True

    return _detected_platform


def _inject_platform(args: str, platform: str) -> str:
    """Append --platform <platform> to args."""
    return f"{args} --platform {platform}"


@tool
async def agent_device(args: str) -> str:
    """
    Controls a mobile device simulator/emulator to navigate, interact, and extract
    data using the 'agent-device' CLI.

    Supports iOS simulators, iOS physical devices, Android emulators, and Android devices.

    Platform is detected automatically on first use by running 'agent-device devices'.
    You do NOT need to specify '--platform' — it is injected for you where required.
    You may still pass '--platform ios' or '--platform android' explicitly to override.

    Common commands:

    - Open app:                 "open Settings"
    - Open URL in browser:      "open 'https://example.com'"
    - Snapshot (interactive):   "snapshot -i"
    - Diff UI changes:          "diff snapshot"
    - Click element:            "click @e1"
    - Semantic find & click:    "find 'Sign In' click"
    - Fill input (clear+type):  "fill @e1 'text'"
    - Type into focused field:  "type 'text'"
    - Scroll:                   "scroll down 0.5"
    - Hardware back (Android):  "back"
    - Home screen:              "home"
    - Wait for element:         "wait @e1" or "wait 3000"
    - Get element text:         "get text @e1"
    - Read clipboard:           "clipboard read"
    - List apps:                "apps"
    - App logs path:            "logs path"
    - Close app:                "close"
    - Close + shutdown device:  "close --shutdown"

    Workflow:
    1. Boot (if no device is ready): "boot"
    2. Open an app: "open MyApp"
    3. Snapshot to get element refs: "snapshot -i"
    4. Interact using refs (@e1, @e2, …) or semantic 'find'.
    5. Re-snapshot after every navigation or UI mutation — refs go stale.

    Session management:
    - Pass '--session <name>' to run multiple isolated device sessions in parallel.
    - Default session is used when '--session' is omitted.

    Args:
        args: Arguments to pass to agent-device CLI.
              Examples: "open Settings", "click @e1", "snapshot -i --session my-session"

    Returns:
        Command output (for observation commands) or a '✓ Done' confirmation
        (for fire-and-forget actions), or an error description.
    """

    effective_args = args
    if _needs_platform(args) and not _has_platform_flag(args):
        platform = await _detect_platform()
        if platform:
            effective_args = _inject_platform(args, platform)

    full_command = f"agent-device {effective_args}"
    print(f"Executing Device Command: {full_command}")

    wait_for_output = _needs_output(effective_args)
    timeout = _get_timeout(effective_args, wait_for_output)

    try:
        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
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
            if not wait_for_output:
                return f"✓ Done: {args}"
            return f"Error: Device command timed out (limit: {timeout}s)."

        output = stdout.decode().strip()
        error_msg = stderr.decode().strip()

        if process.returncode == 0:
            if not wait_for_output:
                return f"✓ Done: {args}"
            return output if output else "Command executed successfully (no output)."
        else:
            return (
                f"Error (Exit Code {process.returncode}):\n"
                f"Stdout: {output}\n"
                f"Stderr: {error_msg}"
            )

    except Exception as e:
        return f"Device Execution Exception: {str(e)}"
