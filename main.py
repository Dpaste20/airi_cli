import asyncio
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.knowledge import Knowledge
from agno.models.llama_cpp import LlamaCpp
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.FileModify import file_modify
from utils.FileSearch import file_search
from utils.FileWrite import file_write
from utils.GetBatteryStatus import get_battery_status
from utils.GetDateTime import get_current_datetime
from utils.GetDiskSpace import get_disk_space
from utils.GetRunningProcesses import get_running_processes
from utils.GetSystemLogs import get_system_logs
from utils.GetUptime import get_uptime
from utils.KillProcess import kill_processes
from utils.OpenApplication import open_application
from utils.RagSearch import get_knowledge_base, initialize_rag, rag_search_tool

logging.getLogger("agno").setLevel(logging.ERROR)

DB_PATH = "tmp/alpha.db"

TOOLS = [
    get_battery_status,
    get_running_processes,
    get_uptime,
    get_disk_space,
    get_system_logs,
    file_search,
    kill_processes,
    rag_search_tool,
    file_write,
    file_modify,
    open_application,
    get_current_datetime,
]

storage_db: Optional[SqliteDb] = None
session_preferences: Dict[str, Dict[str, any]] = {}


def get_session_preference(session_id: str, key: str, default=None):
    return session_preferences.get(session_id, {}).get(key, default)


def set_session_preference(session_id: str, key: str, value):
    if session_id not in session_preferences:
        session_preferences[session_id] = {}
    session_preferences[session_id][key] = value


def parse_command(message: str, session_id: str) -> tuple[str, Optional[str]]:
    message = message.strip()

    if message.startswith("/set "):
        parts = message[5:].strip().split()

        if len(parts) == 0:
            return "", "Usage: /set <preference> <value>\nAvailable: no_think, think"

        command = parts[0].lower()

        if command == "no_think":
            set_session_preference(session_id, "thinking_mode", False)
            return "", " Thinking mode disabled "

        elif command == "think":
            set_session_preference(session_id, "thinking_mode", True)
            return "", " Thinking mode enabled. "

        else:
            return "", f"Unknown preference: {command}\nAvailable: no_think, think"

    if message.startswith("/help"):
        help_text = """Available commands:
• /set no_think
• /set think
• /help

Current settings:
• Thinking mode: {}""".format(
            "enabled"
            if get_session_preference(session_id, "thinking_mode", True)
            else "disabled"
        )
        return "", help_text

    thinking_enabled = get_session_preference(session_id, "thinking_mode", True)

    if not thinking_enabled and not message.endswith("/no_think"):
        message = message + " /no_think"

    return message, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global storage_db
    print("Initializing Airi Backend...")
    storage_db = SqliteDb(db_file=DB_PATH)
    await initialize_rag()
    print("System initialized successfully")
    yield
    print("\nCleaning up session...")
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f"Session database '{DB_PATH}' deleted.")
        except PermissionError:
            print(f"Warning: Could not delete {DB_PATH}.")


app = FastAPI(title="Airi Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent(session_id: str) -> Agent:
    if not storage_db:
        raise ValueError("Database not initialized")

    sys_msg = os.getenv("AGENT_SYSTEM_MESSAGE")
    kb = get_knowledge_base()

    return Agent(
        session_id=session_id,
        model=LlamaCpp(
            id="qwen3:airi",
            temperature=0.0,
            top_p=0.7,
        ),
        system_message=sys_msg,
        db=storage_db,
        knowledge=kb,
        tools=TOOLS,
        search_knowledge=True,
        add_history_to_context=True,
        num_history_runs=10,
        reasoning=False,
        markdown=True,
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"


class ChatResponse(BaseModel):
    response: str


def speak_response(text: str):
    try:
        subprocess.Popen(["spd-say", text])
    except Exception as e:
        print(f"Error executing spd-say: {e}")


@app.get("/")
async def root():
    return {"message": "Airi Agent API is running", "status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        processed_message, command_response = parse_command(
            request.message, request.session_id
        )

        if command_response:
            return ChatResponse(response=command_response)

        local_agent = get_agent(session_id=request.session_id)
        response = await local_agent.arun(processed_message)

        speak_response(response.content)

        return ChatResponse(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    if get_knowledge_base() is None:
        await websocket.send_json({"error": "RAG System not initialized"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            session_id = data.get("session_id", f"ws_{id(websocket)}")

            if not message:
                continue

            try:
                processed_message, command_response = parse_command(message, session_id)

                if command_response:
                    await websocket.send_json({"type": "start"})
                    await websocket.send_json(
                        {"type": "chunk", "content": command_response}
                    )
                    await websocket.send_json(
                        {
                            "type": "end",
                            "token_count": len(command_response) // 4,
                            "generation_time": 0.001,
                        }
                    )
                    continue

                local_agent = get_agent(session_id=session_id)
                await websocket.send_json({"type": "start"})

                start_time = time.perf_counter()
                response_iterator = local_agent.arun(processed_message, stream=True)

                full_response_text = ""
                token_count = 0

                async for chunk in response_iterator:
                    content = ""
                    if hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                    elif isinstance(chunk, str):
                        content = chunk

                    if content:
                        full_response_text += content
                        token_count += len(content) / 4.0
                        await websocket.send_json({"type": "chunk", "content": content})

                generation_time = time.perf_counter() - start_time

                if full_response_text:
                    speak_response(full_response_text)

                await websocket.send_json(
                    {
                        "type": "end",
                        "token_count": int(token_count),
                        "generation_time": generation_time,
                    }
                )

            except Exception as e:
                print(f"Processing error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
