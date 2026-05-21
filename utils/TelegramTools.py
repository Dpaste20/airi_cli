import json
import os

import requests
import yaml

CONFIG_PATH = "tg.yaml"


def send_telegram_message(chat_id: str, message: str) -> str:
    """
    str: Status message indicating success or failure.
    """

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return "Error: TELEGRAM_BOT_TOKEN not found."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return f"Message sent to {chat_id}"
        return f"Failed to send: {response.text}"
    except Exception as e:
        return f"Error sending message: {str(e)}"


def list_telegram_contacts() -> str:
    """
    Returns:
        str: A JSON string of contacts (names and chat_ids).
    """
    if not os.path.exists(CONFIG_PATH):
        return f"Error: {CONFIG_PATH} not found."

    try:
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)

        contacts = config.get("telegram_contacts", [])

        if not contacts:
            return "No contacts found ."

        return json.dumps(contacts, indent=2)

    except yaml.YAMLError as e:
        return f"Error parsing config.yaml: {str(e)}"
    except Exception as e:
        return f"Unexpected error reading contacts: {str(e)}"
