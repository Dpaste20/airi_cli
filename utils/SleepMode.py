import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "SleepMode")


@tool
def sleep_mode_system() -> list:
    if not os.path.exists(BINARY_PATH):
        return [
            {
                "error": f"Binary not found at {BINARY_PATH}. Please compile the Go utility."
            }
        ]

    try:
        subprocess.Popen(
            [BINARY_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        return [
            {
                "status": "success",
                "message": "Sleep mode initiated. System will suspend in 5 seconds.",
            }
        ]

    except Exception as e:
        return [{"error": str(e)}]
