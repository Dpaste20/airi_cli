import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "KillProcess")


@tool
def kill_processes(pid: int, force: bool = False) -> dict:
    if not os.path.exists(BINARY_PATH):
        return {"status": "error", "message": f"Binary not found at {BINARY_PATH}"}

    try:
        cmd = [BINARY_PATH, "-pid", str(pid)]
        if force:
            cmd.append("-force")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        return json.loads(result.stdout)

    except json.JSONDecodeError:
        return {"status": "error", "message": "Failed to decode JSON from Go utility"}
    except subprocess.CalledProcessError as e:
        err_msg = (
            e.stderr.strip() if e.stderr else (e.stdout.strip() if e.stdout else str(e))
        )
        return {"status": "error", "message": f"Go utility failed: {err_msg}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
