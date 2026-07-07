import asyncio
import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import emoji
import tomllib
from agno.agent import Agent
from agno.db.json import JsonDb
from agno.models.ollama import Ollama
from agno.skills import LocalSkills, Skills
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.AdbKeyPress import adb_key_press
from utils.AgentBrowser import agent_browser
from utils.AgentDevice import agent_device
from utils.CalendarTools import (
    create_event,
    delete_event,
    get_upcoming_events,
)
from utils.CameraTools import (
    delete_capture,
    get_recording_status,
    list_captures,
    open_capture,
    start_recording,
    stop_recording,
    take_picture,
    take_timelapse,
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
from utils.GoalTools import (
    add_objective,
    add_sub_goal,
    create_goal,
    delete_goal,
    delete_objective,
    delete_sub_goal,
    edit_goal,
    get_goal,
    list_goals,
    log_goal_progress,
    update_goal_status,
    update_objective_status,
    update_sub_goal_status,
)
from utils.GoogleDriveTools import (
    download_from_drive,
    list_drive_files,
    search_drive_files,
    upload_to_drive,
)
from utils.GoogleTasksTools import add_task, complete_task, delete_task, list_tasks
from utils.KillProcess import kill_processes
from utils.LaunchGames import get_game_list, launch_game
from utils.ManaNetwork import (
    check_mana_agents_status,
    interact_with_mana_network,
    wake_mana_agent,
)
from utils.MapTools import get_directions, map_search
from utils.NotionTools import notion
from utils.OpenApplication import open_application
from utils.OpenUrl import open_url
from utils.PlayMusicTools import (
    list_songs,
    next_song,
    pause_music,
    play_playlist,
    play_random,
    play_song,
    previous_song,
    set_volume,
    stop_music,
)
from utils.RagSearch import get_knowledge_base, initialize_rag, rag_search_tool
from utils.RegionNewsTools import get_region_news, get_top_news, get_topic_news
from utils.RestartSystem import restart_system
from utils.RunDiagnosticTool import run_system_diagnostic

# from utils.Sharedbrowser import shared_browser
from utils.ShellCommandRunner import bash
from utils.Shutdown import shutdown_system
from utils.SkillsTools import install_skill, list_skills
from utils.SleepMode import sleep_mode_system
from utils.SystemInfo import get_system_info
from utils.TelegramTools import list_telegram_contacts, send_telegram_message

logging.getLogger("agno").setLevel(logging.ERROR)
load_dotenv()


DB_PATH = "tmp/alpha_db"


TOOLS = [
    add_objective,
    add_sub_goal,
    create_goal,
    delete_goal,
    delete_objective,
    delete_sub_goal,
    edit_goal,
    get_goal,
    list_goals,
    log_goal_progress,
    update_goal_status,
    update_objective_status,
    update_sub_goal_status,
    # shared_browser,
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
    map_search,
    get_directions,
    list_telegram_contacts,
    send_telegram_message,
    run_system_diagnostic,
    delete_capture,
    get_recording_status,
    list_captures,
    start_recording,
    stop_recording,
    open_capture,
    take_picture,
    take_timelapse,
    launch_game,
    get_game_list,
    get_region_news,
    get_topic_news,
    get_top_news,
    list_songs,
    next_song,
    pause_music,
    play_playlist,
    play_random,
    play_song,
    previous_song,
    set_volume,
    stop_music,
    agent_device,
    adb_key_press,
    notion,
    check_mana_agents_status,
    interact_with_mana_network,
    wake_mana_agent,
    list_skills,
    install_skill,
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


def save_chat_log(
    session_id: str,
    user_message: str,
    agent_response: str,
    attached_files: Optional[List[str]] = None,
):
    log_dir = "logs"

    os.makedirs(log_dir, exist_ok=True)

    today_date = time.strftime("%Y-%m-%d")
    filename = os.path.join(log_dir, f"conversation_{today_date}.log")

    try:
        with open(filename, "a", encoding="utf-8") as f:
            current_time = time.strftime("%H:%M:%S")
            f.write(f"--- [{current_time}] Session: {session_id} ---\n")
            if attached_files:
                f.write(f"Attachments: {', '.join(attached_files)}\n")
            f.write(f"User: {user_message}\n")
            f.write(f"Airi: {agent_response}\n")
            f.write("\n")
    except Exception as e:
        print(f"Error saving chat log: {e}")


def load_config(config_path="config.toml"):
    with open(config_path, "rb") as file:
        return tomllib.load(file)


def get_agent(session_id: str) -> Agent:
    if not storage_db:
        raise ValueError("Database not initialized")

    raw_config = load_config()
    config = AppConfig(**raw_config)

    kb = get_knowledge_base()

    return Agent(
        session_id=session_id,
        model=Ollama(
            id=config.agent.model.id,
            options={"temperature": config.agent.model.temperature},
        ),
        system_message=config.agent.system_message,
        db=storage_db,
        knowledge=kb,
        tools=TOOLS,
        skills=Skills(loaders=[LocalSkills("./skills")]),
        instructions=config.agent.instructions,
        search_knowledge=True,
        add_history_to_context=True,
        num_history_runs=10,
        markdown=True,
    )


class FileAttachment(BaseModel):
    name: str
    content: str
    mime_type: str


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"
    files: List[FileAttachment] = []


class ModelConfig(BaseModel):
    id: str
    temperature: float = 0.3


class AgentConfig(BaseModel):
    model: ModelConfig
    system_message: str
    instructions: List[str]


class WhisperConfig(BaseModel):
    cli_path: str
    model_path: str


class AppConfig(BaseModel):
    agent: AgentConfig
    whisper: WhisperConfig


class ChatResponse(BaseModel):
    response: str


def _extract_pdf_text(raw_bytes: bytes, name: str) -> str:

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"--- Page {i} ---\n{text.strip()}")
        if not pages:
            return (
                f"[PDF '{name}': no extractable text found (may be scanned/image-only)]"
            )
        return "\n\n".join(pages)
    except Exception as exc:
        return f"[PDF '{name}': extraction failed — {exc}]"


def decode_file_content(file: FileAttachment) -> str:
    """Decode a FileAttachment into a plain-text string ready for the prompt."""
    try:
        raw = base64.b64decode(file.content)
    except Exception as exc:
        return f"[File '{file.name}': could not decode base64 — {exc}]"

    if file.mime_type == "application/pdf":
        return _extract_pdf_text(raw, file.name)

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def build_message_with_files(message: str, files: List[FileAttachment]) -> str:
    """
    Prepend file contents to the user message as clearly delimited blocks.

    The agent receives a single string like:

        <file name="report.pdf" type="application/pdf">
        ... extracted text ...
        </file>

        <file name="notes.txt" type="text/plain">
        ... raw text ...
        </file>

        <user_message>
        Summarise the report above.
        </user_message>
    """
    if not files:
        return message

    blocks: List[str] = []
    for f in files:
        body = decode_file_content(f)
        blocks.append(f'<file name="{f.name}" type="{f.mime_type}">\n{body}\n</file>')

    joined = "\n\n".join(blocks)
    return f"{joined}\n\n<user_message>\n{message}\n</user_message>"


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
    """
    Decodes the incoming 44.1kHz audio chunk, downsamples it to 16kHz mono via ffmpeg,
    and runs it locally through whisper-cli.
    """
    raw_config = load_config()
    config = AppConfig(**raw_config)

    WHISPER_CLI_PATH = config.whisper.cli_path
    WHISPER_MODEL_PATH = config.whisper.model_path

    input_filename = ""
    resampled_filename = ""

    try:
        audio_bytes = base64.b64decode(base64_audio)

        with tempfile.NamedTemporaryFile(delete=False, suffix="_raw.wav") as f_in:
            f_in.write(audio_bytes)
            input_filename = f_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix="_16k.wav") as f_out:
            resampled_filename = f_out.name

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_filename,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            resampled_filename,
        ]
        subprocess.run(
            ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        whisper_cmd = [
            WHISPER_CLI_PATH,
            "-m",
            WHISPER_MODEL_PATH,
            "-f",
            resampled_filename,
            "-nt",
        ]
        result = subprocess.run(whisper_cmd, capture_output=True, text=True, check=True)

        transcript_lines = []
        for line in result.stdout.splitlines():
            cleaned_line = line.strip()
            if cleaned_line and not cleaned_line.startswith("["):
                transcript_lines.append(cleaned_line)

        final_text = " ".join(transcript_lines).strip()

        return final_text if final_text else "Could not understand audio"

    except subprocess.CalledProcessError as e:
        print(f"Subprocess conversion execution failed: {e}")
        return "Audio error: external tool failure"
    except Exception as e:
        print(f"Audio processing error: {e}")
        return f"Audio error: {e}"

    finally:
        for path in (input_filename, resampled_filename):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


@app.get("/")
async def root():
    return {"message": "Airi Agent API is running", "status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        local_agent = get_agent(session_id=request.session_id)
        full_message = build_message_with_files(request.message, request.files)
        response = await local_agent.arun(full_message)

        file_names = [f.name for f in request.files] if request.files else None
        save_chat_log(request.session_id, request.message, response.content, file_names)

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

            raw_files = data.get("files") or []
            attached_files: List[FileAttachment] = []
            for rf in raw_files:
                try:
                    attached_files.append(FileAttachment(**rf))
                except Exception as parse_err:
                    print(f"Skipping malformed file attachment: {parse_err}")

            session_id = data.get("session_id", f"ws_{id(websocket)}")

            if not message:
                continue

            if message == "Could not understand audio":
                error_response = "Could not understand audio, would you try again"

                await websocket.send_json({"type": "start"})
                await websocket.send_json({"type": "chunk", "content": error_response})
                await websocket.send_json(
                    {
                        "type": "end",
                        "token_count": len(error_response) // 4,
                        "generation_time": 0,
                    }
                )

                speak_response(error_response)
                save_chat_log(session_id, "Audio unreadable", error_response)

                continue

            try:
                local_agent = get_agent(session_id=session_id)
                await websocket.send_json({"type": "start"})

                start_time = time.perf_counter()

                response_iterator = local_agent.arun(message, stream=True)

                full_response_text = ""
                last_chunk = None

                async for chunk in response_iterator:
                    content = ""
                    if hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                    elif isinstance(chunk, str):
                        content = chunk

                    if content:
                        full_response_text += content
                        await websocket.send_json({"type": "chunk", "content": content})
                    last_chunk = chunk

                generation_time = time.perf_counter() - start_time

                token_count = 0
                if last_chunk and hasattr(last_chunk, "metrics") and last_chunk.metrics:
                    token_count = last_chunk.metrics.total_tokens or 0

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
