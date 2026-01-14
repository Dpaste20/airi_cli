import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "GetActiveConnections")


@tool
def get_active_connections(limit: int = 20, state: str = "") -> list:
    """
    Retrieves a list of active network connections (TCP/UDP) on the system.

    Args:
        limit (int): The maximum number of connections to return (default 20).
        state (str): Optional filter for connection state (e.g., "ESTABLISHED", "LISTEN", "TIME_WAIT").
                     Leave empty to see all states.

    Returns:
        list: A list of dictionaries containing protocol, local/remote address, status, PID, and program name.
    """
    if not os.path.exists(BINARY_PATH):
        return [
            {
                "error": f"Binary not found at {BINARY_PATH}. Please compile the Go utility."
            }
        ]

    try:
        cmd = [BINARY_PATH, "-limit", str(limit)]
        if state:
            cmd.extend(["-state", state])

        result = subprocess.run(
            cmd,
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
