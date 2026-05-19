import asyncio
import os

from agno.tools import tool

OBSERVATION_COMMANDS = (
    # Inspection & Context
    "snapshot",
    "diff",
    "apps",
    "devices",
    "appstate",
    # Element queries / Asserts
    "get ",
    "is ",
    "find ",
    "alert",
    # Debug / Diagnostics / Media
    "perf",
    "metrics",
    "logs",
    "network dump",
    "network log",
    "screenshot",
    "react-devtools",
    # Clipboard / Keyboard
    "clipboard read",
    "keyboard status",
    "keyboard get",
    # Session
    "session",
    # Blocking waits / Suites
    "wait ",
    "batch",
    "test",
    "replay",
)


ADB_KEYMAP = {
    "enter": "66",
    "tab": "61",
    "space": "62",
    "backspace": "67",
    "del": "67",
    "back": "4",
    "home": "3",
    "menu": "82",
    "power": "26",
    "volup": "24",
    "voldown": "25",
    "up": "19",
    "down": "20",
    "left": "21",
    "right": "22",
}


def _needs_output(args: str) -> bool:
    """Returns True if the command produces output that must be read."""
    stripped = args.strip().lower()
    return any(stripped.startswith(cmd) for cmd in OBSERVATION_COMMANDS)


def _resolve_timeout(args: str, wait_for_output: bool) -> int:
    """
    Returns the appropriate timeout in seconds.

    - wait/batch/test/replay/react-devtools: 180s (can be long-running)
    - Other observation commands: 120s
    - Fire-and-forget actions: 30s
    """
    stripped = args.strip().lower()
    if any(
        stripped.startswith(cmd)
        for cmd in ("wait ", "batch", "test", "replay", "react-devtools")
    ):
        return 180
    return 120 if wait_for_output else 30


async def _run_single(args: str, env: dict) -> tuple[bool, str]:
    """
    Runs a single agent-device or adb command.
    Returns (success, output_string).
    """
    stripped_args = args.strip()
    wait_for_output = _needs_output(stripped_args)
    timeout = _resolve_timeout(stripped_args, wait_for_output)

    if stripped_args.lower().startswith("keyevent "):
        key = stripped_args.split(" ", 1)[1].strip().lower()
        keycode = ADB_KEYMAP.get(key, key)
        full_command = f"adb shell input keyevent {keycode}"
        print(f"Executing ADB Command: {full_command}")
    elif stripped_args.lower().startswith("adb "):
        full_command = stripped_args
        print(f"Executing Raw ADB Command: {full_command}")
    else:
        full_command = f"agent-device {stripped_args}"
        print(f"Executing Device Command: {full_command}")

    try:
        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
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
                return True, f"✓ Done: {args}"
            return False, f"Error: '{args}' timed out (limit: {timeout}s)."

        output = stdout.decode().strip()
        error_msg = stderr.decode().strip()

        if process.returncode == 0:
            if not wait_for_output:
                return True, f"✓ Done: {args}"
            return (
                True,
                output if output else "Command executed successfully (no output).",
            )
        else:
            return False, (
                f"Error (Exit Code {process.returncode}):\n"
                f"Stdout: {output}\n"
                f"Stderr: {error_msg}"
            )

    except Exception as e:
        return False, f"Device Execution Exception: {str(e)}"


@tool
async def agent_device(args: str) -> str:
    """
    Controls an emulator, simulator, or physical device to automate mobile (iOS/Android), TV, and macOS desktop apps.

    CRITICAL — BATCHING RULE
    ------------------------
    NEVER call this tool once per command. ALWAYS pass all sequential commands
    together in a single call, separated by newlines. The tool executes them in
    order and returns combined output.

    The ONLY reason to make a second call is when you need to read snapshot
    output (e.g. element refs like @e1) before deciding what to interact with next.

    WORKFLOW
    --------
    1. Check installed apps: "apps --platform ios" (or android)
    2. Open an app:          "open <app_name> --platform ios"
    3. Snapshot elements:    "snapshot -i"
    4. Interact via refs:    "click @e1", "fill @e2 'text'"
    5. Re-snapshot after any screen/UI mutation.

    HARDWARE KEYEVENTS & ADB (Android Only)
    ---------------------------------------
    You can trigger native Android hardware events using 'keyevent'.
    - "keyevent enter"           Press the Enter key
    - "keyevent tab"             Press the Tab key
    - "keyevent back"            Press the hardware Back button
    - "keyevent home"            Press the hardware Home button
    - "keyevent 66"              Use raw Android keycodes if needed
    - "adb shell input tap 100 100" Run raw ADB commands natively

    NAVIGATION & SESSIONS
    ---------------------
    - "boot --platform ios"      Ensure target is ready
    - "open <app|url>"           Launch an app or deep link
    - "back"                     App-owned back UI
    - "home"                     Go to home screen
    - "close"                    Close active app
    - "rotate portrait"          Rotate orientation

    SNAPSHOTS & VISION
    ------------------
    - "snapshot -i"              Interactive elements only (RECOMMENDED). Provides @eN refs.
    - "snapshot"                 Full accessibility tree
    - "diff snapshot"            Compare structural changes against previous step
    - "screenshot page.png"      Take a screenshot

    INTERACTION (use @eN refs from snapshot)
    ----------------------------------------
    - "click @e1"                Click/tap element
    - "fill @e1 'text'"          Clear the input AND type text into it
    - "type 'text'"              Type into currently focused element (does not clear)
    - "press @e1"                Press down on an element
    - "longpress @x @y 800"      Long press coordinates for 800ms
    - "swipe x1 y1 x2 y2 250"    Swipe gesture
    - "scroll down 0.5"          Scroll half a screen down
    - "scroll down --pixels 300" Scroll fixed distance
    - "keyboard dismiss"         Hide the on-screen keyboard

    SEMANTIC FINDERS
    ----------------
    - "find 'Sign In' click"             Find text and click
    - "find label 'Email' fill 'test@'"  Find element by label and fill
    - "find role button click"           Find semantic role

    GET INFO & ASSERTIONS
    ---------------------
    - "get text @e1"             Get text content of element
    - "is visible 'label=\"Continue\"'" Evaluate if selector is visible
    - "wait 1500"                Wait for milliseconds
    - "wait @e1"                 Wait for a known element to appear

    DEBUGGING & MEDIA
    -----------------
    - "perf"                     Get CPU, memory, and frame-health/startup metrics
    - "logs clear --restart"     Clear app logs and start tailing
    - "network dump 25"          Dump last 25 HTTP requests from logs
    - "react-devtools status"    Check React Native connection
    - "react-devtools get tree"  Get React component tree

    Args:
        args (str): Arguments to pass to agent-device CLI (e.g., "open Settings --platform ios").
                    Supports multiple newline-separated commands.
    """
    env = os.environ.copy()

    commands = [
        line.strip()
        for line in args.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not commands:
        return "Error: No commands provided."

    if len(commands) == 1:
        _, result = await _run_single(commands[0], env)
        return result

    results: list[str] = []
    for i, cmd in enumerate(commands, 1):
        success, output = await _run_single(cmd, env)
        results.append(f"[{i}] {cmd}\n{output}")
        if not success:
            remaining = len(commands) - i
            if remaining:
                results.append(f"⚠ Aborted: {remaining} command(s) not executed.")
            break

    return "\n\n".join(results)
