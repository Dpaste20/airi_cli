import base64
import os
from email.mime.text import MIMEText
from typing import List, Optional

from agno.tools import tool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Define the scope for reading and modifying (to mark as read/apply labels)
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
TOKEN_FILE = "tmp/gmail_token.json"


def get_gmail_service():
    """
    Handles the OAuth2 authentication flow using environment variables.
    """
    creds = None

    # Load existing token if available
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # If credentials don't exist or are invalid, initiate login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Dynamically build the config from your provided environment variables
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
            # This will open a browser window for local authentication
            creds = flow.run_local_server(port=0)

        # Save the token for subsequent runs to avoid re-auth
        os.makedirs("tmp", exist_ok=True)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


@tool
async def get_unread_emails(max_results: int = 5) -> str:
    """
    Retrieves unread emails from the inbox.
    Returns the sender, subject, and a brief snippet.
    """
    try:
        service = get_gmail_service()
        # Query: 'is:unread' filters for unread, 'label:inbox' ensures it's in the main folder
        results = (
            service.users()
            .messages()
            .list(userId="me", q="is:unread label:inbox", maxResults=max_results)
            .execute()
        )

        messages = results.get("messages", [])
        if not messages:
            return "No unread emails found."

        email_list = []
        for msg in messages:
            # Fetch the full content for each message ID
            m = service.users().messages().get(userId="me", id=msg["id"]).execute()
            headers = m.get("payload", {}).get("headers", [])

            # Extract specific header info
            subject = next(
                (h["value"] for h in headers if h["name"] == "Subject"), "No Subject"
            )
            sender = next(
                (h["value"] for h in headers if h["name"] == "From"), "Unknown"
            )
            snippet = m.get("snippet", "")

            email_list.append(
                f"ID: {msg['id']}\nFROM: {sender}\nSUBJ: {subject}\nSNIPPET: {snippet}\n{'-' * 20}"
            )

        return "\n".join(email_list)

    except Exception as e:
        return f"Error accessing Gmail: {str(e)}"


@tool
async def search_emails(query: str, max_results: int = 5) -> str:
    """
    Searches for emails using standard Gmail query syntax.

    Args:
        query: The search query (e.g., "from:alice", "is:read", "subject:hello").
        max_results: The maximum number of emails to return (default 5).

    Returns:
        A formatted string containing the sender, subject, and snippet of matching emails.
    """
    try:
        service = get_gmail_service()
        # list() uses the same 'q' parameter you would type into the Gmail search bar
        results = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )

        messages = results.get("messages", [])
        if not messages:
            return f"No emails found matching query: '{query}'"

        email_list = []
        for msg in messages:
            # Fetch full message details
            m = service.users().messages().get(userId="me", id=msg["id"]).execute()
            headers = m.get("payload", {}).get("headers", [])

            subject = next(
                (h["value"] for h in headers if h["name"] == "Subject"), "No Subject"
            )
            sender = next(
                (h["value"] for h in headers if h["name"] == "From"), "Unknown"
            )
            snippet = m.get("snippet", "")

            email_list.append(
                f"ID: {msg['id']}\nFROM: {sender}\nSUBJ: {subject}\nSNIPPET: {snippet}\n{'-' * 20}"
            )

        return "\n".join(email_list)

    except Exception as e:
        return f"Error searching Gmail: {str(e)}"


@tool
async def send_email_reply(thread_id: str, body: str) -> str:
    """
    Sends a reply to the specified email thread.
    It automatically finds the last message in the thread to reply to the correct sender
    and maintains the conversation history (threading).

    Args:
        thread_id: The ID of the thread to reply to.
        body: The text content of the reply.

    Returns:
        A success message with the new message ID or an error description.
    """
    try:
        service = get_gmail_service()

        # 1. Fetch the thread to get the latest message details
        # We need this to know who to reply to and what the subject is
        thread = service.users().threads().get(userId="me", id=thread_id).execute()
        messages = thread.get("messages", [])

        if not messages:
            return "Error: Thread not found or empty."

        # Get the last message in the thread (the one we are replying to)
        last_msg = messages[-1]
        last_msg_id = last_msg["id"]

        # Fetch full headers for the last message
        msg_detail = (
            service.users().messages().get(userId="me", id=last_msg_id).execute()
        )
        headers = msg_detail.get("payload", {}).get("headers", [])

        # Extract necessary headers to form a valid reply
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "")
        message_id_header = next(
            (h["value"] for h in headers if h["name"] == "Message-ID"), ""
        )
        references = next(
            (h["value"] for h in headers if h["name"] == "References"), ""
        )

        # 2. Construct the MIME Message
        mime_message = MIMEText(body)

        # 'To' should be the sender of the last message
        mime_message["To"] = sender

        # Handle Subject: Ensure it starts with "Re:" if it doesn't already
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        mime_message["Subject"] = subject

        # CRITICAL: These headers tell Gmail this is a reply in the same thread
        mime_message["In-Reply-To"] = message_id_header
        mime_message["References"] = f"{references} {message_id_header}".strip()

        # 3. Encode and Send
        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()

        create_message = {
            "raw": encoded_message,
            "threadId": thread_id,  # Explicitly link to the thread ID
        }

        sent_message = (
            service.users().messages().send(userId="me", body=create_message).execute()
        )

        return f"Reply sent successfully. New Message ID: {sent_message['id']}"

    except Exception as e:
        return f"Error sending reply: {str(e)}"


@tool
async def create_draft_email(to: str, subject: str, body: str) -> str:
    """
    Creates a draft email in Gmail without sending it.
    Useful for reviewing content before valid dispatch.

    Args:
        to: The recipient's email address.
        subject: The email subject line.
        body: The main content of the email.

    Returns:
        A success message with the Draft ID.
    """
    try:
        service = get_gmail_service()

        # Construct the MIME message
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        # Encode the message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        # The structure for drafts is {'message': {'raw': ...}}
        create_body = {"message": {"raw": raw_message}}

        draft = service.users().drafts().create(userId="me", body=create_body).execute()

        return f"Draft created successfully. Draft ID: {draft['id']}"

    except Exception as e:
        return f"Error creating draft: {str(e)}"


@tool
async def send_email(to: str, subject: str, body: str) -> str:
    """
    Sends a fresh email to a specified recipient.

    Args:
        to: The recipient's email address.
        subject: The subject of the email.
        body: The content of the email.

    Returns:
        A success message with the Message ID.
    """
    try:
        service = get_gmail_service()

        # Construct the MIME message
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        # Encode the message for Gmail API (base64url)
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"raw": raw_message}

        sent_message = (
            service.users().messages().send(userId="me", body=create_message).execute()
        )

        return f"Email sent successfully. Message ID: {sent_message['id']}"

    except Exception as e:
        return f"Error sending email: {str(e)}"
