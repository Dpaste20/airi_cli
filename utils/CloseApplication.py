import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "CloseApplication")


@tool
def close_application(application_name: str) -> dict:
    """
    Closes a running application by its name.
    Use this to stop programs like 'Chrome', 'Discord', or 'VS Code'.

    Args:
        application_name (str): The common name or process name of the app to close.
    """
    if not os.path.exists(BINARY_PATH):
        return {
            "error": f"Binary not found at {BINARY_PATH}. Compile the Go code first."
        }

    try:
        result = subprocess.run(
            [BINARY_PATH, application_name], capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)

    except subprocess.CalledProcessError:
        return {
            "error": f"Failed to close '{application_name}'. It might not be running."
        }
    except Exception as e:
        return {"error": str(e)}
