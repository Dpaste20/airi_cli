import os
from typing import Optional

from agno.tools import tool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/tasks"]
TOKEN_FILE = "tmp/tasks_token.json"


def get_tasks_service():
    """
    Handles OAuth2 authentication for Google Tasks.
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_config = {
                "installed": {
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "project_id": os.getenv("GOOGLE_PROJECT_ID"),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": [
                        os.getenv("GOOGLE_REDIRECT_URI", "http://localhost")
                    ],
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)

        os.makedirs("tmp", exist_ok=True)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("tasks", "v1", credentials=creds)


@tool
async def list_tasks(max_results: int = 10) -> str:
    """
    Lists pending (incomplete) tasks from the user's default task list.
    Returns: Task Title, ID, and Due Date (if any).
    """
    try:
        service = get_tasks_service()

        results = (
            service.tasks()
            .list(tasklist="@default", maxResults=max_results, showCompleted=False)
            .execute()
        )

        items = results.get("items", [])

        if not items:
            return "No pending tasks found."

        task_list = []
        for item in items:
            title = item.get("title", "(No Title)")
            task_id = item.get("id")
            due = item.get("due", "No due date")

            if due != "No due date":
                due = due[:10]

            task_list.append(f"Task: {title}\nID: {task_id}\nDue: {due}\n{'-' * 20}")

        return "\n".join(task_list)

    except Exception as e:
        return f"Error listing tasks: {str(e)}"


@tool
async def add_task(title: str, notes: str = "", due_date: Optional[str] = None) -> str:
    """
    Adds a new task to the default task list.

    Args:
        title: The main text of the task.
        notes: (Optional) Description or extra details.
        due_date: (Optional) Due date in RFC 3339 format (e.g., "2023-10-25T00:00:00.000Z").
    """
    try:
        service = get_tasks_service()

        task_body = {"title": title, "notes": notes}

        if due_date:
            task_body["due"] = due_date

        result = service.tasks().insert(tasklist="@default", body=task_body).execute()

        return f"Task created successfully.\nID: {result.get('id')}\nLink: {result.get('selfLink')}"

    except Exception as e:
        return f"Error creating task: {str(e)}"


@tool
async def complete_task(task_id: str) -> str:
    """
    Marks a task as completed.

    Args:
        task_id: The ID of the task to complete (get this from list_tasks).
    """
    try:
        service = get_tasks_service()

        task = service.tasks().get(tasklist="@default", task=task_id).execute()

        task["status"] = "completed"

        updated_task = (
            service.tasks()
            .update(tasklist="@default", task=task_id, body=task)
            .execute()
        )

        return f"Task '{updated_task.get('title')}' marked as completed."

    except Exception as e:
        return f"Error completing task: {str(e)}"


@tool
async def delete_task(task_id: str) -> str:
    """
    Permanently deletes a task.

    Args:
        task_id: The ID of the task to delete.
    """
    try:
        service = get_tasks_service()
        service.tasks().delete(tasklist="@default", task=task_id).execute()
        return f"Task {task_id} deleted successfully."
    except Exception as e:
        return f"Error deleting task: {str(e)}"
