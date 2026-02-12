import json
import os
import subprocess
from typing import Optional

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "CronManager", "CronManager")
CRON_LOG_DIR = os.path.join(os.getcwd(), "Cron Job")


def _run_cron_manager(args: list) -> str:
    """Helper to run the Go binary and return stdout."""
    if not os.path.exists(BINARY_PATH):
        return json.dumps(
            {"error": f"Binary not found at {BINARY_PATH}. Please compile it."}
        )

    try:
        result = subprocess.run(
            [BINARY_PATH] + args, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        return json.dumps({"error": f"CronManager failed: {error_msg}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _save_job_details(job_id: str, schedule: str, command: str):
    """Saves the cron job details to a local JSON file."""
    try:
        if not os.path.exists(CRON_LOG_DIR):
            os.makedirs(CRON_LOG_DIR)

        file_path = os.path.join(CRON_LOG_DIR, f"{job_id}.json")

        job_data = {
            "id": job_id,
            "schedule": schedule,
            "command": command,
            "created_at": os.popen("date").read().strip(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=4)

    except Exception as e:
        print(f"Failed to save cron job log: {e}")


@tool
def get_cron_jobs() -> str:
    """
    Retrieves the list of currently scheduled cron jobs.
    Returns a JSON list containing the ID, Schedule, and Command for each job.
    """
    return _run_cron_manager(["list"])


@tool
def add_cron_job(schedule: str, command: str) -> str:
    """
    Adds a new cron job to the system schedule and saves a backup log.

    Args:
        schedule: The cron schedule expression (e.g., "*/5 * * * *" or "0 9 * * 1").
        command: The full command to execute. MUST use absolute paths.

    Returns:
        A JSON string indicating success or failure.
    """

    response_str = _run_cron_manager(["add", schedule, command])

    try:
        response = json.loads(response_str)

        if response.get("status") == "success" and "id" in response:
            _save_job_details(response["id"], schedule, command)

    except json.JSONDecodeError:
        pass

    return response_str


@tool
def delete_cron_job(job_id: str) -> str:
    """
    Deletes a cron job using its ID and removes its local log file.

    Args:
        job_id: The unique 8-character ID of the job.

    Returns:
        A JSON string indicating success or failure.
    """
    response_str = _run_cron_manager(["remove", job_id])

    try:
        response = json.loads(response_str)
        if response.get("status") == "success":
            log_file = os.path.join(CRON_LOG_DIR, f"{job_id}.json")
            if os.path.exists(log_file):
                os.remove(log_file)
    except:
        pass

    return response_str
