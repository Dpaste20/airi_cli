import subprocess

from agno.tools import tool


@tool
def kill_processes(pid: int, force: bool = False) -> dict:
    """
    Terminates a process identified by its PID.

    Args:
        pid (int): The process ID of the process to terminate.
        force (bool): If True, forces the process to terminate immediately (SIGKILL). Default is False.
    """
    try:
        cmd = ["kill"]

        if force:
            cmd.append("-9")

        cmd.append(str(pid))

        # Run the command, capturing stderr to report specific errors if it fails
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        return {
            "status": "success",
            "message": f"Process {pid} {'forcefully ' if force else ''}terminated.",
        }

    except subprocess.CalledProcessError as e:
        # This handles cases like "Operation not permitted" or "No such process"
        error_msg = e.stderr.strip() if e.stderr else "Unknown error occurred"
        return {
            "status": "error",
            "message": f"Failed to kill process {pid}: {error_msg}",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
