import asyncio
import logging
import os
import random

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.ollama import Ollama
from agno.tools import tool
from agno.vectordb.qdrant import Qdrant
from dotenv import load_dotenv

from utils.asciiArt import ascii_art

logging.getLogger("agno").setLevel(logging.ERROR)
load_dotenv()
DB_PATH = "tmp/alpha.db"


async def run_agent():
    db = SqliteDb(db_file=DB_PATH)

    vector_db = Qdrant(
        collection="airi_knowledge",
        url="http://localhost:6333",
        embedder=OllamaEmbedder(id="nomic-embed-text:v1.5", dimensions=768),
    )

    knowledge_base = Knowledge(vector_db=vector_db)

    await knowledge_base.add_content_async(path="tmp/test_sample.pdf")

    sys_description = os.getenv("AGENT_SYSTEM_INSTRUCTION")

    agent = Agent(
        model=Ollama(id="llama3.2:1b"),
        description=sys_description,
        db=db,
        knowledge=knowledge_base,
        search_knowledge=False,
        # tools=[get_weather],
        session_id="browser_session",
        add_history_to_context=True,
        num_history_runs=10,
        markdown=True,
    )

    while True:
        try:
            user_command = input("\nEnter your command (or 'exit' to quit): ")

            if user_command.lower() == "exit":
                break

            if "@search_knowledge" in user_command:
                agent.search_knowledge = True

                user_command = user_command.replace("@search_knowledge", "").strip()
            else:
                agent.search_knowledge = False

            await agent.aprint_response(user_command, stream=True)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        pass
    finally:
        print("\nCleaning up session...")
        if os.path.exists(DB_PATH):
            try:
                os.remove(DB_PATH)
                print(f"Session database '{DB_PATH}' deleted.")
            except PermissionError:
                print(f"Warning: Could not delete {DB_PATH} (file might be in use).")
        else:
            print("No session database found to delete.")
