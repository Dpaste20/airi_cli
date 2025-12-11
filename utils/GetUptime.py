import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "GetUptime")


@tool
def get_uptime() -> dict:
    """
    Retrieves system uptime by reading /proc/uptime using the compiled Go utility.
    """
    if not os.path.exists(BINARY_PATH):
        return {"error": f"Binary not found at {BINARY_PATH}"}

    try:
        result = subprocess.run(
            [BINARY_PATH],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from Go utility"}
    except subprocess.CalledProcessError as e:
        return {"error": f"Go utility failed: {e}"}
    except Exception as e:
        return {"error": str(e)}
