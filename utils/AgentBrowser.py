import asyncio
import os

from agno.tools import tool

BROWSER_DEBUG_PORT = int(os.getenv("BO_DEBUG_PORT", 9222))
BROWSER_HOST = os.getenv("BO_DEBUG_HOST", "127.0.0.1")


async def is_browser_running(host=BROWSER_HOST, port=BROWSER_DEBUG_PORT) -> bool:
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


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

    if not await is_browser_running():
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

            print(
                f"Port ready! Running 'agent-browser connect {BROWSER_DEBUG_PORT}'..."
            )
            connect_process = await asyncio.create_subprocess_shell(
                f"agent-browser connect {BROWSER_DEBUG_PORT}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _, connect_stderr = await connect_process.communicate()

            if connect_process.returncode != 0:
                return f"Error connecting agent-browser to port {BROWSER_DEBUG_PORT}: {connect_stderr.decode().strip()}"

            print("Connection successful.")

        except Exception as e:
            return f"Error launching browser: {str(e)}"

    full_command = f"agent-browser {args}"
    print(f"Executing Browser Command: {full_command}")

    try:
        process = await asyncio.create_subprocess_shell(
            full_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
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
            return f"Error (Exit Code {process.returncode}):\nStdout: {output}\nStderr: {error_msg}"

    except Exception as e:
        return f"Browser Execution Exception: {str(e)}"
