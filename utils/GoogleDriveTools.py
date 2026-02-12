import base64
import io
import mimetypes
import os
from typing import Optional

from agno.tools import tool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = "tmp/drive_token.json"


def get_drive_service():
    """
    Handles the OAuth2 authentication flow using environment variables.
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

    return build("drive", "v3", credentials=creds)


@tool
async def list_drive_files(max_results: int = 10) -> str:
    """
    Lists files from the root of Google Drive.
    Returns file names, IDs, and types.
    """
    try:
        service = get_drive_service()
        results = (
            service.files()
            .list(
                pageSize=max_results, fields="nextPageToken, files(id, name, mimeType)"
            )
            .execute()
        )
        items = results.get("files", [])

        if not items:
            return "No files found."

        file_list = []
        for item in items:
            file_list.append(
                f"Name: {item['name']} | ID: {item['id']} | Type: {item['mimeType']}"
            )

        return "\n".join(file_list)

    except Exception as e:
        return f"Error listing Drive files: {str(e)}"


@tool
async def search_drive_files(query: str) -> str:
    """
    Searches for files in Google Drive.
    Args:
        query: The search text (matches file name).
    """
    try:
        service = get_drive_service()

        q = f"name contains '{query}' and trashed = false"

        results = (
            service.files()
            .list(q=q, pageSize=10, fields="nextPageToken, files(id, name, mimeType)")
            .execute()
        )
        items = results.get("files", [])

        if not items:
            return f"No files found matching '{query}'"

        file_list = []
        for item in items:
            file_list.append(
                f"Name: {item['name']} | ID: {item['id']} | Type: {item['mimeType']}"
            )

        return "\n".join(file_list)

    except Exception as e:
        return f"Error searching Drive: {str(e)}"


@tool
async def upload_to_drive(file_path: str, drive_folder_id: Optional[str] = None) -> str:
    """
    Uploads a local file to Google Drive.

    Args:
        file_path: The local path to the file to upload.
        drive_folder_id: (Optional) The ID of the Drive folder to upload into.
                         If not provided, uploads to root.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    try:
        service = get_drive_service()
        file_name = os.path.basename(file_path)

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        file_metadata = {"name": file_name}
        if drive_folder_id:
            file_metadata["parents"] = [drive_folder_id]

        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )

        return f"File uploaded successfully. File ID: {file.get('id')}"

    except Exception as e:
        return f"Error uploading to Drive: {str(e)}"


@tool
async def download_from_drive(file_id: str, local_path: str) -> str:
    """
    Downloads a file from Google Drive to a local path.

    Args:
        file_id: The ID of the file on Google Drive.
        local_path: The local path (including filename) where the file should be saved.
    """
    try:
        service = get_drive_service()
        request = service.files().get_media(fileId=file_id)

        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

        fh = io.FileIO(local_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()

        return f"File downloaded successfully to {local_path}"

    except Exception as e:
        return f"Error downloading from Drive: {str(e)}"
