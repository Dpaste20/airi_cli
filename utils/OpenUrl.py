import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "OpenUrl")


@tool
def open_url(url: str) -> dict:
    """
    Opens a URL in the system's default web browser.

    Args:
        url (str): The URL to open. Protocol (https://) will be added automatically if not present.

    Returns:
        dict: A dictionary containing the opened URL and success message, or an error message.
              Example: {"url": "https://...", "message": "URL opened successfully"} or {"error": "..."}
    """
    if not os.path.exists(BINARY_PATH):
        return {
            "error": f"Binary not found at {BINARY_PATH}. Please compile the Go utility."
        }

    if not url or not url.strip():
        return {"error": "URL cannot be empty"}

    try:
        cmd = [BINARY_PATH, "-url", url]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        return json.loads(result.stdout)

    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from Go utility"}
    except subprocess.CalledProcessError as e:
        return {"error": f"Go utility execution failed: {e}"}
    except Exception as e:
        return {"error": str(e)}
