import asyncio
import shlex

from agno.tools import tool


@tool
async def bash(command: str) -> str:
    """
    Executes a shell command asynchronously to prevent blocking the FastAPI server.
    """
    print(f"Executing Shell Command: {command}")

    try:
        # Create the subprocess asynchronously
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        # Wait for the command to finish with a timeout (e.g., 2 minutes)
        # This yields control back to the event loop while waiting
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except asyncio.TimeoutError:
            # If it takes too long, kill it to free resources
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return "Error: Command execution timed out (limit: 120s)."

        output = stdout.decode().strip()
        error_msg = stderr.decode().strip()

        if process.returncode == 0:
            return output if output else "Command executed successfully (no output)."
        else:
            return f"Error (Exit Code {process.returncode}): {error_msg}"

    except Exception as e:
        return f"Execution Exception: {str(e)}"
