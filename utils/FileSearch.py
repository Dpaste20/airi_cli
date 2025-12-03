import logging
import subprocess

from agno.tools import tool


@tool
def file_search(query: str) -> str:
    try:
        # print(f"DEBUG: Deep searching system for '*{query}*'...")

        command = ["find", "/", "-type", "f", "-iname", f"*{query}*"]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        found_files = result.stdout.strip()
        if not found_files:
            return f"No files found matching '*{query}*' in the entire system."

        file_list = found_files.split("\n")
        count = len(file_list)
        if count > 20:
            return f"Found {count} files. Here are the first 20:\n" + "\n".join(
                file_list[:20]
            )
        return f"Found the following files:\n{found_files}"
    except subprocess.TimeoutExpired:
        return "Search timed out. Searching the entire root filesystem took too long. Try a more specific name."
    except Exception as e:
        logging.error(f"Search tool error: {e}")
        return f"An error occurred: {str(e)}"
