import asyncio
import base64
import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import emoji
import speech_recognition as sr
from agno.agent import Agent
from agno.db.json import JsonDb
from agno.models.ollama import Ollama
from agno.skills import LocalSkills, Skills
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.AgentBrowser import agent_browser
from utils.CalendarTools import (
    create_event,
    delete_event,
    get_upcoming_events,
)
from utils.CronTools import add_cron_job, delete_cron_job, get_cron_jobs
from utils.FetchUrls import fetch_urls
from utils.FileModify import file_modify
from utils.FileSearch import file_search
from utils.FileWrite import file_write
from utils.GetActiveConnections import get_active_connections
from utils.GetBatteryStatus import get_battery_status
from utils.GetDateTime import get_current_datetime
from utils.GetDiskSpace import get_disk_space
from utils.GetIPInfo import get_ip_info
from utils.GetRunningProcesses import get_running_processes
from utils.GetSystemLogs import get_system_logs
from utils.GetUptime import get_uptime
from utils.GmailTools import (
    create_draft_email,
    get_unread_emails,
    search_emails,
    send_email,
    send_email_reply,
)
from utils.GoogleDriveTools import (
    download_from_drive,
    list_drive_files,
    search_drive_files,
    upload_to_drive,
)
from utils.GoogleTasksTools import add_task, complete_task, delete_task, list_tasks
from utils.KillProcess import kill_processes
from utils.OpenApplication import open_application
from utils.OpenUrl import open_url
from utils.RagSearch import get_knowledge_base, initialize_rag, rag_search_tool
from utils.RestartSystem import restart_system
from utils.ShellCommandRunner import bash
from utils.Shutdown import shutdown_system
from utils.SleepMode import sleep_mode_system
from utils.SystemInfo import get_system_info

logging.getLogger("agno").setLevel(logging.ERROR)
load_dotenv()

DB_PATH = "tmp/alpha_db"


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
    get_ip_info,
    get_active_connections,
    get_system_info,
    shutdown_system,
    sleep_mode_system,
    restart_system,
    fetch_urls,
    open_url,
    bash,
    get_unread_emails,
    search_emails,
    send_email_reply,
    create_draft_email,
    send_email,
    get_cron_jobs,
    add_cron_job,
    delete_cron_job,
    list_drive_files,
    search_drive_files,
    upload_to_drive,
    download_from_drive,
    get_upcoming_events,
    create_event,
    delete_event,
    list_tasks,
    add_task,
    complete_task,
    delete_task,
    agent_browser,
]

storage_db: Optional[JsonDb] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global storage_db
    print("Initializing Airi Backend...")
    storage_db = JsonDb(db_path=DB_PATH)
    await initialize_rag()
    print("System initialized successfully")
    yield
    print("\nCleaning up session...")

    stop_audio()

    if os.path.exists(DB_PATH):
        try:
            if os.path.isdir(DB_PATH):
                shutil.rmtree(DB_PATH)
            else:
                os.remove(DB_PATH)

            print(f"Session database '{DB_PATH}' deleted.")
        except PermissionError:
            print(f"Warning: Could not delete {DB_PATH}.")
        except Exception as e:
            print(f"Error deleting database: {e}")


app = FastAPI(title="Airi Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_chat_log(session_id: str, user_message: str, agent_response: str):

    log_dir = "logs"

    os.makedirs(log_dir, exist_ok=True)

    today_date = time.strftime("%Y-%m-%d")
    filename = os.path.join(log_dir, f"conversation_{today_date}.log")

    try:
        with open(filename, "a", encoding="utf-8") as f:
            current_time = time.strftime("%H:%M:%S")
            f.write(f"--- [{current_time}] Session: {session_id} ---\n")
            f.write(f"User: {user_message}\n")
            f.write(f"Airi: {agent_response}\n")
            f.write("\n")
    except Exception as e:
        print(f"Error saving chat log: {e}")


def get_agent(session_id: str) -> Agent:
    if not storage_db:
        raise ValueError("Database not initialized")

    sys_msg = os.getenv("AGENT_SYSTEM_MESSAGE")
    kb = get_knowledge_base()

    return Agent(
        session_id=session_id,
        model=Ollama(id="glm-5:cloud"),
        system_message=sys_msg,
        db=storage_db,
        knowledge=kb,
        tools=TOOLS,
        skills=Skills(loaders=[LocalSkills("./skills/agent-browser")]),
        instructions=[
            "You have access to specialized skills.",
            "Use get_skill_instructions to load full guidance when needed.",
            "CRITICAL: Before using create_event, get_upcoming_events, or add_task, YOU MUST ALWAYS call get_current_datetime to get the current date and time.",
            "When converting the date from get_current_datetime (DD/MM/YYYY) for Google APIs, ensure you convert it to YYYY-MM-DD format.",
        ],
        search_knowledge=True,
        add_history_to_context=True,
        num_history_runs=10,
        markdown=True,
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"


class ChatResponse(BaseModel):
    response: str


def speak_response(text: str):
    stop_audio()

    clean_text = emoji.replace_emoji(text, replace="")

    try:
        subprocess.Popen(["spd-say", clean_text])
    except Exception as e:
        print(f"Error executing spd-say: {e}")


def stop_audio():
    try:
        subprocess.run(["spd-say", "-C"], check=False)
    except Exception as e:
        print(f"Error stopping audio: {e}")


def transcribe_audio(base64_audio: str) -> str:
    r = sr.Recognizer()
    temp_filename = ""
    try:
        audio_bytes = base64.b64decode(base64_audio)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_filename = f.name

        with sr.AudioFile(temp_filename) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except Exception as e:
        return f"Audio error: {e}"
    finally:
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass


@app.get("/")
async def root():
    return {"message": "Airi Agent API is running", "status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        local_agent = get_agent(session_id=request.session_id)
        response = await local_agent.arun(request.message)

        save_chat_log(request.session_id, request.message, response.content)

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

            if data.get("action") == "stop_speech":
                stop_audio()
                await websocket.send_json({"type": "speech_stopped"})
                continue

            message = ""
            if "audio_data" in data and data["audio_data"]:
                print("Receiving audio data...")
                transcribed_text = transcribe_audio(data["audio_data"])
                print(f"Transcribed: {transcribed_text}")

                await websocket.send_json(
                    {"type": "chunk", "content": f"🎤 *Voice:* {transcribed_text}\n\n"}
                )
                message = transcribed_text
            else:
                message = data.get("message", "")

            session_id = data.get("session_id", f"ws_{id(websocket)}")

            if not message or message == "Could not understand audio":
                continue

            try:
                local_agent = get_agent(session_id=session_id)
                await websocket.send_json({"type": "start"})

                start_time = time.perf_counter()

                response_iterator = local_agent.arun(message, stream=True)

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

                    save_chat_log(session_id, message, full_response_text)

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
        stop_audio()
    except Exception as e:
        print(f"WebSocket error: {e}")
        stop_audio()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
