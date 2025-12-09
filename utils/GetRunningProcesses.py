import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "GetRunningProcesses")


@tool
def get_running_processes(limit: int = 10) -> list:
    """
    Returns a list of the top running processes sorted by CPU usage using a Go utility.

    Args:
        limit (int): The number of processes to return (default is 10).
    """
    if not os.path.exists(BINARY_PATH):
        return [
            {
                "error": f"Binary not found at {BINARY_PATH}. Please compile the Go utility."
            }
        ]

    try:
        # Call the Go binary
        result = subprocess.run(
            [BINARY_PATH, "-limit", str(limit)],
            capture_output=True,
            text=True,
            check=True,
        )

        return json.loads(result.stdout)

    except json.JSONDecodeError:
        return [{"error": "Failed to decode JSON from Go utility"}]
    except subprocess.CalledProcessError as e:
        return [{"error": f"Go utility execution failed: {e}"}]
    except Exception as e:
        return [{"error": str(e)}]
