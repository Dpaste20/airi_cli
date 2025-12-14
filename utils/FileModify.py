import json
import os
import subprocess

from agno.tools import tool

# Path to the compiled Go binary
BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "FileModify")
STORAGE_DIR = os.path.join(os.getcwd(), "Airi_created_files")


@tool
def file_modify(filename: str, old_text: str, new_text: str) -> str:
    """
    Modifies an existing file by replacing a specific text snippet with new text.
    Use this for minor edits to avoid rewriting the whole file.

    Args:
        filename (str): The relative name of the file inside 'Airi_created_files' (e.g., "notes.txt").
        old_text (str): The exact text snippet to find and remove.
        new_text (str): The new text to insert in its place.
    """
    if not os.path.exists(BINARY_PATH):
        return f"Error: Binary not found at {BINARY_PATH}. Compile it first!"

    full_path = os.path.join(STORAGE_DIR, filename)

    if not os.path.exists(full_path):
        return f"Error: File '{filename}' does not exist."

    try:
        result = subprocess.run(
            [BINARY_PATH, "-path", full_path, "-old", old_text, "-new", new_text],
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)

        if data.get("error"):
            return f"Error: {data['error']}"

        return "Success: File updated successfully."

    except subprocess.CalledProcessError as e:
        return f"System Error: {e}"
    except json.JSONDecodeError:
        return "Error: Failed to parse tool output."
    except Exception as e:
        return f"Modify failed: {str(e)}"
