import asyncio
import os
import shutil

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.ollama import Ollama
from dotenv import load_dotenv

load_dotenv()


async def run_agent():
    db = SqliteDb(db_file="tmp/alpha.db")

    sys_description = os.getenv("AGENT_SYSTEM_INSTRUCTION")

    agent = Agent(
        model=Ollama(id="gemma3:1b"),
        description=sys_description,
        db=db,
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

            await agent.aprint_response(user_command, stream=True)

        except Exception as e:
            print(f"Error: {e}")


def print_centered_art(file_path):
    try:
        terminal_width = shutil.get_terminal_size().columns
    except OSError:
        terminal_width = 80

    with open(file_path, "r") as file:
        content = file.read()

    lines = content.splitlines()

    max_width = 0
    if lines:
        max_width = max(len(line) for line in lines)

    padding = (terminal_width - max_width) // 2

    padding_str = " " * max(0, padding)

    print("\n")
    for line in lines:
        print(f"{padding_str}{line}")
    print("\n")


print_centered_art("ascii_art.txt")


asyncio.run(run_agent())
