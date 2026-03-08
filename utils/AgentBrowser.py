import asyncio
import os

from agno.tools import tool

_session_started: bool = False

OBSERVATION_COMMANDS = (
    "snapshot",
    "screenshot",
    "eval",
    "get ",
    "is ",
    "cookies get",
    "storage",
    "state list",
    "state show",
    "session",
    "tab list",
    "diff",
)


def _needs_output(args: str) -> bool:
    """Returns True if the command needs to wait for output."""
    stripped = args.strip().lower()
    return any(stripped.startswith(cmd) for cmd in OBSERVATION_COMMANDS)


async def _close_stale_session(env: dict) -> None:
    """Closes any stale agent-browser daemon/session from previous runs."""
    print("Closing any stale agent-browser session...")
    try:
        process = await asyncio.create_subprocess_shell(
            "agent-browser close",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        await asyncio.wait_for(process.communicate(), timeout=10)
    except Exception:
        pass


@tool
async def agent_browser(args: str) -> str:
    """
    Controls a web browser to navigate, interact, and extract data using the 'agent-browser' CLI.

    Runs in native mode (direct CDP via Rust binary) with a persistent profile and headed
    (visible) browser window. Configured via environment variables injected per command:
      AGENT_BROWSER_NATIVE=1
      AGENT_BROWSER_PROFILE=./airi_browse_dir

    Common commands:
    - Open URL: "open <url>"
    - Snapshot (get elements): "snapshot -i"
    - Click element: "click @e1" (use ID from snapshot)
    - Fill input: "fill @e1 'text'"
    - Type text: "type @e1 'text'"
    - Scroll: "scroll down 500"
    - Go back: "back"
    - Wait: "wait 5000"

    Workflow:
    1. Open a URL.
    2. Take a snapshot to get element IDs (@e1, @e2...).
    3. Interact using those IDs.
    4. Snapshot again if the page changes.

    Args:
        args (str): The arguments to pass to the agent-browser CLI (e.g., "open google.com" or "click @e1").
    """
    global _session_started

    env = os.environ.copy()
    env["AGENT_BROWSER_NATIVE"] = "1"
    env["AGENT_BROWSER_PROFILE"] = os.path.abspath("airi_browse_dir")

    if not _session_started:
        await _close_stale_session(env)
        _session_started = True

    full_command = f"agent-browser --headed {args}"
    print(f"Executing Browser Command: {full_command}")

    wait_for_output = _needs_output(args)

    try:
        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        timeout = 120 if wait_for_output else 15
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
            return "Error: Browser command timed out (limit: 120s)."

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
        return f"Browser Execution Exception: {str(e)}"
