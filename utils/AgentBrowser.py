import asyncio

from agno.tools import tool


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
