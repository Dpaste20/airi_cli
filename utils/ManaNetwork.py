import asyncio
import json
import os

import websockets
from agno.tools import tool

MANA_WS_URL = os.getenv("MANA_WS_URL")


async def _run_mana_ws_command(payload: dict) -> str:
    """Helper function to send a command to the MANA Hub and stream the response."""
    if not MANA_WS_URL:
        return (
            "Error: MANA_WS_URL environment variable is not set. Please configure it."
        )
    try:
        async with websockets.connect(MANA_WS_URL) as ws:
            await ws.send(json.dumps(payload))

            full_response = ""
            while True:
                resp = await ws.recv()
                data = json.loads(resp)

                if data.get("type") == "chunk":
                    full_response += data.get("content", "")
                elif data.get("type") == "error":
                    return f"Error from MANA: {data.get('message')}"
                elif data.get("type") == "end":
                    break

            return full_response.strip()
    except Exception as e:
        return f"Failed to connect to MANA Hub at {MANA_WS_URL}: {str(e)}"


@tool
async def check_mana_agents_status() -> str:
    """
    Check the online/offline status of all AI agents in the MANA network.
    Use this to see if peers (like Zephyr, ITAA, or KUBER) are available for delegation.
    """
    payload = {"action": "check_online", "agents": ["all"]}
    return await _run_mana_ws_command(payload)


@tool
async def wake_mana_agent(agent_slug: str) -> str:
    """
    Start up a dormant agent in the MANA network.
    Pass the exact slug of the agent (e.g., 'zephyr', 'itta', 'kuber').
    """
    payload = {"action": "wake_agent", "agents": [agent_slug]}
    return await _run_mana_ws_command(payload)


@tool
async def interact_with_mana_network(message: str) -> str:
    """
    Send a message to the MANA network to collaborate with or delegate tasks to other AI agents.
    CRITICAL: You MUST include an @mention for the target agent in your message
    so the Hub can route it (e.g., '@zephyr analyze this data' or '@all status report').
    """
    payload = {"action": "chat", "message": message}
    return await _run_mana_ws_command(payload)
