import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "FileWrite")
STORAGE_DIR = os.path.join(os.getcwd(), "Airi_created_files")


@tool
def file_write(filename: str, content: str, append: bool = False) -> str:
    if not os.path.exists(BINARY_PATH):
        return f"Error: Binary not found at {BINARY_PATH}. Compile it first!"

    full_path = os.path.join(STORAGE_DIR, filename)

    cmd = [BINARY_PATH, "-path", full_path, "-content", content]
    if append:
        cmd.append("-append")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        if data.get("error"):
            return f"Error: {data['error']}"

        return (
            f"Success: {data.get('message')} (Saved to: Airi_created_files/{filename})"
        )

    except subprocess.CalledProcessError as e:
        return f"System Error: Process failed. {e}"
    except json.JSONDecodeError:
        return "Error: Failed to parse tool output."
    except Exception as e:
        return f"Write failed: {str(e)}"
