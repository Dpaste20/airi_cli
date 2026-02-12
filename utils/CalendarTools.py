import datetime
import os
from typing import Optional

from agno.tools import tool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_FILE = "tmp/calendar_token.json"


def get_calendar_service():
    """
    Handles OAuth2 authentication for Google Calendar.
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

    return build("calendar", "v3", credentials=creds)


@tool
async def get_upcoming_events(max_results: int = 10) -> str:
    """
    Gets the next upcoming events from the user's primary calendar.
    """
    try:
        service = get_calendar_service()
        now = datetime.datetime.utcnow().isoformat() + "Z"

        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])

        if not events:
            return "No upcoming events found."

        event_list = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event.get("summary", "(No Title)")
            event_id = event.get("id")
            event_list.append(f"Time: {start} | Event: {summary} | ID: {event_id}")

        return "\n".join(event_list)

    except Exception as e:
        return f"Error fetching calendar events: {str(e)}"


@tool
async def create_event(
    summary: str, start_time: str, end_time: str, description: str = ""
) -> str:
    """
    Creates a new calendar event.

    Args:
        summary: Title of the event.
        start_time: Start time in ISO format (e.g., '2023-10-25T14:00:00').
        end_time: End time in ISO format (e.g., '2023-10-25T15:00:00').
        description: (Optional) Description of the event.
    """
    try:
        service = get_calendar_service()

        event = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": "UTC",
            },
            "end": {
                "dateTime": end_time,
                "timeZone": "UTC",
            },
        }

        event = service.events().insert(calendarId="primary", body=event).execute()
        return f"Event created: {event.get('htmlLink')}"

    except Exception as e:
        return f"Error creating event: {str(e)}"


@tool
async def delete_event(event_id: str) -> str:
    """
    Deletes a calendar event by ID.

    Args:
        event_id: The unique ID of the event to delete (retrievable via get_upcoming_events).
    """
    try:
        service = get_calendar_service()
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return f"Event {event_id} deleted successfully."
    except Exception as e:
        return f"Error deleting event: {str(e)}"
