import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "RunDiagnostic")


@tool
def run_system_diagnostic() -> dict:
    """
    Runs a system diagnostic check retrieving concurrent system metrics including:
    - Overall System Status (Healthy/Warning)
    - CPU Usage (Load Average & Cores)
    - RAM Usage (Used/Total and Percentage)
    - Disk Space (Free/Total and Percentage)
    - Thermals (Current Temperature in °C)
    - Network Ping (Connectivity speed)

    Returns:
        dict: A dictionary containing the diagnostic report or an error message.
    """
    if not os.path.exists(BINARY_PATH):
        return {
            "error": f"Binary not found at {BINARY_PATH}. Please compile the Go utility."
        }

    try:
        # Call the Go binary
        result = subprocess.run(
            [BINARY_PATH],
            capture_output=True,
            text=True,
            check=True,
        )

        return json.loads(result.stdout)

    except json.JSONDecodeError:
        return {
            "error": "Failed to decode JSON from Go utility. Ensure the Go script outputs valid JSON."
        }
    except subprocess.CalledProcessError as e:
        return {"error": f"Go utility execution failed: {e}"}
    except Exception as e:
        return {"error": str(e)}
