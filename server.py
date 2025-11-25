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
from agno.vectordb.qdrant import Qdrant
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.GetBatteryStatus import get_battery_status
from utils.GetRunningProcesses import get_running_processes

logging.getLogger("agno").setLevel(logging.ERROR)
load_dotenv()

DB_PATH = "tmp/alpha.db"

knowledge_base: Optional[Knowledge] = None
storage_db: Optional[SqliteDb] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global knowledge_base, storage_db

    print("Initializing system...")

    storage_db = SqliteDb(db_file=DB_PATH)

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    vector_db = Qdrant(
        collection="airi_knowledge",
        url=qdrant_url,
        embedder=OllamaEmbedder(id="nomic-embed-text:v1.5", dimensions=768),
    )

    knowledge_base = Knowledge(vector_db=vector_db)

    pdf_path = "tmp/test_sample.pdf"
    try:
        if os.path.exists(pdf_path):
            await knowledge_base.add_content_async(path=pdf_path)
            print("Knowledge base loaded from PDF.")
        else:
            print(f"Warning: PDF not found at {pdf_path}, skipping ingestion.")
    except Exception as e:
        print(f"Warning: Issue loading PDF: {e}")

    print("System initialized successfully")
    yield

    print("\nCleaning up session...")
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f"Session database '{DB_PATH}' deleted.")
        except PermissionError:
            print(f"Warning: Could not delete {DB_PATH} (file might be in use).")


app = FastAPI(title="Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent(session_id: str, search_knowledge: bool) -> Agent:
    if not storage_db or not knowledge_base:
        raise ValueError("Database or Knowledge Base not initialized")

    sys_description = os.getenv(
        "AGENT_SYSTEM_INSTRUCTION", "You are a helpful assistant."
    )

    return Agent(
        session_id=session_id,
        model=Gemini(id="gemini-flash-latest"),
        description=sys_description,
        db=storage_db,
        knowledge=knowledge_base,
        tools=[get_battery_status, get_running_processes],
        search_knowledge=search_knowledge,
        add_history_to_context=True,
        num_history_runs=10,
        markdown=True,
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"
    search_knowledge: bool = False


class ChatResponse(BaseModel):
    response: str
    search_knowledge_used: bool


@app.get("/")
async def root():
    return {"message": "Agent API is running", "status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not knowledge_base:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        local_agent = get_agent(
            session_id=request.session_id, search_knowledge=request.search_knowledge
        )

        response = await local_agent.arun(request.message)

        return ChatResponse(
            response=response.content, search_knowledge_used=request.search_knowledge
        )
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
            search_knowledge = data.get("search_knowledge", False)
            session_id = data.get("session_id", f"ws_{id(websocket)}")

            if not message:
                continue

            try:
                local_agent = get_agent(
                    session_id=session_id, search_knowledge=search_knowledge
                )

                await websocket.send_json(
                    {"type": "start", "search_knowledge_used": search_knowledge}
                )

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
