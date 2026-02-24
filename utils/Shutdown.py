import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "Shutdown")


@tool
def shutdown_system() -> list:
    """
    Initiates a full system shutdown after a 3-second delay.
    """
    if not os.path.exists(BINARY_PATH):
        return [
            {
                "error": f"Binary not found at {BINARY_PATH}. Please compile the Go utility."
            }
        ]

    try:
        # Popen spawns the process in the background without blocking Python
        subprocess.Popen(
            [BINARY_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detaches the Go process from Airi
        )

        # Immediately return the acknowledgement to the agent context
        return [
            {
                "status": "success",
                "message": "Airi and system shutdown initiated. System will power off in 3 seconds.",
            }
        ]

    except Exception as e:
        return [{"error": str(e)}]
