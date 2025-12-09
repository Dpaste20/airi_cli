import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "FileSearch")


@tool
def file_search(query: str) -> str:
    """
    Searches for files matching the query string in the system.
    Skips system directories (/proc, /sys) for speed.

    Args:
        query (str): The filename or part of the filename to search for.
    """
    if not os.path.exists(BINARY_PATH):
        return f"Error: Binary not found at {BINARY_PATH}"

    try:
        result = subprocess.run(
            [BINARY_PATH, "-query", query, "-root", "/", "-timeout", "10"],
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)

        files = data.get("files", [])
        error = data.get("error", "")
        count = data.get("count", 0)

        response = ""
        if error:
            response += f"Note: {error}\n"

        if not files:
            return response + f"No files found matching '{query}'."

        response += f"Found {count} files:\n" + "\n".join(files)
        return response

    except json.JSONDecodeError:
        return "Error: Failed to parse search results."
    except Exception as e:
        return f"Search failed: {str(e)}"
