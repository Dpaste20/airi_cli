import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "RestartMode")


@tool
def restart_system() -> list:
    """
    Initiates a full system restart (reboot) after a 5-second delay.
    """
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
                "message": "Airi and system restart initiated. System will reboot in 5 seconds.",
            }
        ]

    except Exception as e:
        return [{"error": str(e)}]
