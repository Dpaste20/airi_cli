import asyncio
import os

from agno.tools import tool

BROWSER_DEBUG_PORT = int(os.getenv("BO_DEBUG_PORT", 9222))
BROWSER_HOST = os.getenv("BO_DEBUG_HOST", "127.0.0.1")


_agent_browser_connected: bool = False


async def is_browser_running(host=BROWSER_HOST, port=BROWSER_DEBUG_PORT) -> bool:
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def _launch_browser() -> str | None:
    """
    Launches Brave with the remote debugging port open.
    Returns an error string on failure, or None on success.
    """
    print(f"Browser not detected. Launching Brave on port {BROWSER_DEBUG_PORT}...")
    try:
        user_data_dir = os.path.abspath("airi_browse_dir")

        await asyncio.create_subprocess_exec(
            "brave-browser",
            f"--remote-debugging-port={BROWSER_DEBUG_PORT}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-default-apps",
            "--disable-popup-blocking",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        print("Waiting for debugging port to initialize...")
        port_ready = False
        for _ in range(15):
            await asyncio.sleep(1)
            if await is_browser_running():
                port_ready = True
                break

        if not port_ready:
            return f"Error: Brave launched, but port {BROWSER_DEBUG_PORT} did not open within 15 seconds."

        print("Port is open. Waiting for DevTools Protocol to stabilize...")
        await asyncio.sleep(3)

    except Exception as e:
        return f"Error launching browser: {str(e)}"

    return None


async def _connect_agent_browser() -> str | None:
    """
    Runs 'agent-browser connect <port>' to register the debugging port.
    Returns an error string on failure, or None on success.
    """
    global _agent_browser_connected

    print(f"Running 'agent-browser connect {BROWSER_DEBUG_PORT}'...")
    try:
        connect_process = await asyncio.create_subprocess_shell(
            f"agent-browser connect {BROWSER_DEBUG_PORT}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            connect_process.communicate(), timeout=15
        )
    except asyncio.TimeoutError:
        return "Error: 'agent-browser connect' timed out."
    except Exception as e:
        return f"Error running agent-browser connect: {str(e)}"

    if connect_process.returncode != 0:
        error_msg = stderr.decode().strip()
        return (
            f"Error connecting agent-browser to port {BROWSER_DEBUG_PORT}: {error_msg}"
        )

    print("agent-browser connected successfully.")
    _agent_browser_connected = True
    return None


async def _ensure_ready() -> str | None:
    """
    Ensures the browser is running and agent-browser is connected.
    Returns an error string if anything fails, or None if ready.
    """
    global _agent_browser_connected

    browser_was_running = await is_browser_running()

    # Launch browser if not already up
    if not browser_was_running:
        _agent_browser_connected = False  # Browser restarted — must reconnect
        error = await _launch_browser()
        if error:
            return error

    # Connect agent-browser if not yet connected this session,
    # OR if the browser was not running (just launched or restarted).
    if not _agent_browser_connected or not browser_was_running:
        error = await _connect_agent_browser()
        if error:
            return error

    return None


@tool
async def agent_browser(args: str) -> str:
    """
    Controls a web browser to navigate, interact, and extract data using the 'agent-browser' CLI.

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

    error = await _ensure_ready()
    if error:
        return error

    full_command = f"agent-browser {args}"
    print(f"Executing Browser Command: {full_command}")

    try:
        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return "Error: Browser command timed out (limit: 120s)."

        output = stdout.decode().strip()
        error_msg = stderr.decode().strip()

        if process.returncode == 0:
            return output if output else "Command executed successfully (no output)."
        else:
            if "connect" in error_msg.lower() or "not connected" in error_msg.lower():
                global _agent_browser_connected
                _agent_browser_connected = False

            return (
                f"Error (Exit Code {process.returncode}):\n"
                f"Stdout: {output}\n"
                f"Stderr: {error_msg}"
            )

    except Exception as e:
        return f"Browser Execution Exception: {str(e)}"
