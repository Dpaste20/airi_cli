import asyncio
import os
import shlex
import shutil
import tempfile
import urllib.error
import urllib.request

from agno.tools import tool

SHARED_BROWSER_PORT = int(os.getenv("SHARED_BROWSER_PORT", "9222"))

LAUNCH_BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "launch_browser")

_connected: bool = False


OBSERVATION_COMMANDS = (
    "snapshot",
    "screenshot",
    "pdf ",
    "eval ",
    "get ",
    "is ",
    "find ",
    "cookies",
    "storage",
    "network requests",
    "network request ",
    "network har",
    "state list",
    "state show",
    "session",
    "tab",
    "auth list",
    "auth show",
    "console",
    "errors",
    "stream status",
    "profiles",
    "doctor",
    "wait ",
    "batch",
    "clipboard read",
    "react ",
    "vitals",
)


def _needs_output(args: str) -> bool:
    """Returns True if the command produces output that must be read."""
    stripped = args.strip().lower()
    return any(stripped.startswith(cmd) for cmd in OBSERVATION_COMMANDS)


def _resolve_timeout(args: str, wait_for_output: bool) -> int:
    """Returns the appropriate timeout in seconds."""
    stripped = args.strip().lower()
    if any(stripped.startswith(cmd) for cmd in ("wait ", "batch", "doctor")):
        return 180
    return 120 if wait_for_output else 15


def _is_port_live(port: int, timeout: float = 1.5) -> bool:
    """Checks whether a CDP-compatible browser is already listening."""
    url = f"http://localhost:{port}/json/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


async def _launch_shared_browser(port: int) -> tuple[bool, str]:
    """Starts the human's shared browser instance via launch_browser.go."""
    if not os.path.exists(LAUNCH_BINARY_PATH):
        return False, (
            f"Cannot auto-launch: binary not found at {LAUNCH_BINARY_PATH}. "
            f"Compile launch_browser.go first, or start manually with --remote-debugging-port={port}."
        )

    try:
        process = await asyncio.create_subprocess_exec(
            LAUNCH_BINARY_PATH,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

        if process.returncode != 0:
            err = stderr.decode().strip() or stdout.decode().strip()
            return False, f"launch_browser exited with error: {err}"

        return True, stdout.decode().strip() or "Shared browser launched."
    except asyncio.TimeoutError:
        return False, "Timed out waiting for launch_browser to start the browser."
    except Exception as e:
        return False, f"Failed to run launch_browser: {e}"


async def _ensure_connected(port: int) -> tuple[bool, str]:
    """Ensures a CDP session is available on `port`."""
    global _connected

    if _is_port_live(port):
        _connected = True
        return True, ""

    launched, msg = await _launch_shared_browser(port)
    if not launched:
        return False, msg

    for _ in range(5):
        if _is_port_live(port):
            _connected = True
            return True, ""
        await asyncio.sleep(1)

    return False, f"launch_browser ran but port {port} isn't responding."


async def _run_single_shared(args: str, port: int) -> tuple[bool, str]:
    """Runs a single shared browser command using the tempfile daemon logic."""
    global _connected

    try:
        argv = ["agent-browser", "--auto-connect", *shlex.split(args)]
    except ValueError as e:
        return False, f"Error: could not parse command arguments ({e})."

    print(f"Executing Shared Browser Command: {' '.join(argv)}")

    wait_for_output = _needs_output(args)
    timeout = _resolve_timeout(args, wait_for_output)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout_path = os.path.join(tmpdir, "stdout.log")
            stderr_path = os.path.join(tmpdir, "stderr.log")

            with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
                process = await asyncio.create_subprocess_exec(
                    *argv, stdout=out_f, stderr=err_f
                )

                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                        await process.wait()
                    except ProcessLookupError:
                        pass
                    if not wait_for_output:
                        return True, f"✓ Done: {args}"
                    return (
                        False,
                        f"Error: Shared browser command timed out (limit: {timeout}s).",
                    )

            with open(stdout_path, "rb") as f:
                output = f.read().decode(errors="replace").strip()
            with open(stderr_path, "rb") as f:
                error_msg = f.read().decode(errors="replace").strip()

        if process.returncode == 0:
            if not wait_for_output:
                return True, f"✓ Done: {args}"
            return (
                True,
                output if output else "Command executed successfully (no output).",
            )
        else:
            if not _is_port_live(port):
                _connected = False
                return False, (
                    f"Error: the shared browser session appears to have ended. "
                    f"The human may have closed the window. Try again to relaunch."
                )
            return (
                False,
                f"Error (Exit Code {process.returncode}):\nStdout: {output}\nStderr: {error_msg}",
            )

    except Exception as e:
        return False, f"Shared Browser Execution Exception: {str(e)}"


@tool
async def shared_browser(args: str) -> str:
    """
    Controls the SAME browser window the human is actively using, via a shared CDP connection.
    Use this when asked to act in "my browser", "this tab", or "what I'm looking at".

    CRITICAL GUARDRAILS:
    1. NEVER close tabs ("tab close") unless the human explicitly asks you to.
    2. NEVER navigate away from the current page ("open <url>") if the human is actively reading or working on it, unless instructed. Open a new tab instead ("tab new <url>").
    3. BATCH COMMANDS: Pass sequential commands separated by newlines in a single tool call.

    WORKFLOW:
    1. Check tabs ("tab") or take a snapshot ("snapshot -i") to understand current context.
    2. Interact using element IDs (@e1, @e2) from the snapshot.
    3. Re-snapshot after navigation/DOM changes, as IDs go stale.

    NAVIGATION & TABS:
    - "open <url>"               Navigate current tab to URL
    - "tab new <url>"            Open URL in a new tab (SAFE DEFAULT)
    - "tab"                      List all open tabs
    - "tab <id/label>"           Switch to specific tab
    - "back" / "forward"         Browser history

    SNAPSHOTS (Mandatory for Interaction):
    - "snapshot -i"              Returns accessibility tree with element refs (@e1, @e2)
    - "screenshot"               Takes a screenshot
    - "screenshot --annotate"    Screenshot with numbered element labels

    INTERACTION (Requires @eN refs from snapshot):
    - "click @e1"                Click element
    - "click @e1 --new-tab"      Click link and open in new tab
    - "fill @e1 'text'"          Clear and fill input
    - "type @e1 'text'"          Type into element
    - "press Enter"              Press keyboard key
    - "scroll down 500"          Scroll page

    SEMANTIC FINDERS (When refs are unavailable):
    - "find text 'Login' click"
    - "find role button click --name 'Submit'"
    - "find placeholder 'Search' fill 'query'"

    GET INFO & STATE:
    - "get text @e1"             Extract text content
    - "get url"                  Get current URL
    - "get title"                Get page title
    - "is visible @e1"           Check visibility
    - "clipboard read"           Read human's clipboard

    WAITING:
    - "wait 3000"                Wait milliseconds
    - "wait @e1"                 Wait for element to appear
    - "wait --load networkidle"  Wait for network to settle

    Args:
        args (str): Arguments to pass to the CLI. Supports multiple commands
                    separated by newlines (e.g., "tab new google.com\nsnapshot -i").
    """
    global _connected
    port = SHARED_BROWSER_PORT

    if not shutil.which("agent-browser"):
        return "Error: 'agent-browser' CLI not found on PATH."

    if not _connected or not _is_port_live(port):
        ready, msg = await _ensure_connected(port)
        if not ready:
            return f"Error: could not establish shared browser session. {msg}"

        connect_proc = await asyncio.create_subprocess_exec(
            "agent-browser",
            "--auto-connect",
            "connect",
            str(port),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(connect_proc.communicate(), timeout=15)

    commands = [
        line.strip()
        for line in args.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not commands:
        return "Error: No commands provided."

    if len(commands) == 1:
        _, result = await _run_single_shared(commands[0], port)
        return result

    results: list[str] = []
    for i, cmd in enumerate(commands, 1):
        success, output = await _run_single_shared(cmd, port)
        results.append(f"[{i}] {cmd}\n{output}")
        if not success:
            remaining = len(commands) - i
            if remaining:
                results.append(f"⚠ Aborted: {remaining} command(s) not executed.")
            break

    return "\n\n".join(results)
