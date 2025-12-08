import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.google import Gemini
from agno.tools import tool
from agno.vectordb.qdrant import Qdrant
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.FileSearch import file_search
from utils.GetBatteryStatus import get_battery_status
from utils.GetDiskSpace import get_disk_space
from utils.GetRunningProcesses import get_running_processes
from utils.GetSystemLogs import get_system_logs
from utils.GetUptime import get_uptime
from utils.KillProcess import kill_processes
from utils.RagSearch import rag_search

logging.getLogger("agno").setLevel(logging.ERROR)
load_dotenv()

DB_PATH = "tmp/alpha.db"

TOOLS = [
    get_battery_status,
    get_running_processes,
    get_uptime,
    get_disk_space,
    get_system_logs,
    file_search,
    rag_search,
    kill_processes,
]

knowledge_base: Optional[Knowledge] = None
storage_db: Optional[SqliteDb] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global knowledge_base, storage_db

    print("Initializing system...")

    storage_db = SqliteDb(db_file=DB_PATH)

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = "airi_knowledge"

    vector_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        embedder=OllamaEmbedder(id="embeddinggemma:latest", dimensions=768),
    )
    knowledge_base = Knowledge(vector_db=vector_db)

    # --- FIX: Set this to True to force adding new files ---
    should_ingest = True

    # Optional: Print current status, but don't stop ingestion
    try:
        if vector_db.client.collection_exists(collection_name):
            info = vector_db.client.get_collection(collection_name)
            print(f"Current DB status: {info.points_count} existing vectors.")
    except Exception as e:
        print(f"Qdrant status check skipped: {e}")

    if should_ingest:
        # Define your files and metadata here
        documents = [
            {
                "path": "tmp/test_sample.pdf",
                "metadata": {"subject": "PS", "batch": 2026},
            },
            {
                "path": "tmp/test_sample2.pdf",
                "metadata": {"subject": "DP", "batch": 2026},
            },
        ]

        for doc in documents:
            path = doc["path"]
            meta = doc["metadata"]

            if os.path.exists(path):
                print(f"Ingesting {path} with metadata {meta}...")
                # The upsert logic in Agno usually handles duplicates,
                # but this ensures the file is processed.
                await knowledge_base.add_content_async(path=path, metadata=meta)
                print(f"DONE: {path}")
            else:
                print(f"SKIPPING: File not found at {path}")

    print("System initialized successfully")
    yield

    print("\nCleaning up session...")
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f"Session database '{DB_PATH}' deleted.")
        except PermissionError:
            print(f"Warning: Could not delete {DB_PATH}.")


app = FastAPI(title="Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent(session_id: str) -> Agent:
    if not storage_db or not knowledge_base:
        raise ValueError("Database or Knowledge Base not initialized")

    sys_msg = os.getenv("AGENT_SYSTEM_MESSAGE")

    return Agent(
        session_id=session_id,
        model=Gemini(id="gemini-2.5-flash"),
        system_message=sys_msg,
        db=storage_db,
        knowledge=knowledge_base,
        tools=TOOLS,
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


@app.get("/")
async def root():
    return {"message": "Agent API is running", "status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not knowledge_base:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        local_agent = get_agent(session_id=request.session_id)

        response = await local_agent.arun(request.message)

        return ChatResponse(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    if knowledge_base is None:
        print("Error: Knowledge base is None")
        await websocket.send_json({"error": "System not initialized"})
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
                local_agent = get_agent(session_id=session_id)

                await websocket.send_json({"type": "start"})

                response_iterator = local_agent.arun(message, stream=True)

                async for chunk in response_iterator:
                    content = ""
                    if hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                    elif isinstance(chunk, str):
                        content = chunk

                    if content:
                        await websocket.send_json({"type": "chunk", "content": content})

                await websocket.send_json({"type": "end"})

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
