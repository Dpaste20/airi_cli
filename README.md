# Airi — Local-First AI Agent Harness

> Your personal AI. Your hardware. Your rules.

Airi is a fully autonomous AI agent that runs entirely on your machine. It is not a chatbot wrapper, not a cloud API client, not a plugin system. It is a self-contained agent ecosystem with direct, low-level access to your system — capable of reading kernel metrics, controlling mobile device simulators, composing emails, playing music, scraping the web, and executing arbitrary shell commands.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Tool Ecosystem](#tool-ecosystem)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Design Principles](#design-principles)
- [Extending Airi](#extending-airi)

---

## Overview

Airi is built around one conviction: **your personal AI should run on your hardware, answer to you alone, and have real access to your machine** — not a sandboxed approximation of it.

Key properties at a glance:

- **Local-first** — defaults to fully offline execution via Ollama. No data leaves your machine unless a tool explicitly sends it somewhere.
- **Model-agnostic** — swap between 40+ local or cloud models (Ollama, vLLM, OpenAI, Anthropic, DeepSeek, Groq, and more) without touching any tool code.
- **60+ tools** — spanning system management, browser automation, mobile device control, Google Workspace, shell execution, music, camera, news, and more.
- **Open-ended extensibility** — any function decorated with `@tool` becomes a new capability the agent can use immediately.
- **[Watch the demo](https://youtu.be/b8vDnNZmaCo)** 
---

## Architecture

```
┌─────────────────────────────────────────┐
│         Go Bubbletea TUI Frontend        │
│    (persistent WebSocket connection)     │
└────────────────────┬────────────────────┘
                     │ WebSocket / REST
┌────────────────────▼────────────────────┐
│         FastAPI Python Backend           │
│  ┌─────────────────────────────────┐    │
│  │      agno Agent Orchestrator    │    │
│  │  ┌──────────┐  ┌─────────────┐ │    │
│  │  │  Model   │  │  RAG / KB   │ │    │
│  │  │          │  │             │ │    │
│  │  └──────────┘  └─────────────┘ │    │
│  │          Tool Registry          │    │
│  └──────────────┬──────────────────┘    │
└─────────────────┼───────────────────────┘
                  │
     ┌────────────▼────────────┐
     │   Go Compiled Binaries  │  ← system metrics, file I/O,
     │   Python Async Tools    │    browser, mobile, shell, etc.
     └─────────────────────────┘
```

- **Backend:** FastAPI + Python asyncio. Exposes a streaming WebSocket (`/ws/chat`) and a REST endpoint (`/chat`).
- **Frontend:** Go + Bubbletea TUI. Connects over WebSocket and renders streamed responses in real time.
- **Agent Orchestration:** [agno](https://github.com/agno-agi/agno) framework, fully decoupled from the underlying model.
- **Voice:** Speech-to-text via `speech_recognition`, text-to-speech via `spd-say`.
- **Memory:** Session history is ephemeral (wiped on shutdown). Long-term memory persists across conversations in a separate path.
- **RAG Pipeline:** Qdrant + Ollama embeddings. Documents are ingested at startup with MD5 change detection — unchanged files are never re-embedded.
- **Go Utilities:** Compiled binaries in `go-utils/` handle concurrent system metrics, `/proc`/`/sys` reads, and filesystem traversal. Used surgically, not universally.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Go 1.21+
- [Ollama](https://ollama.ai) (for local model execution)
- [Qdrant](https://qdrant.tech) running locally (for RAG)

### 1. Clone and install

```bash
git clone https://github.com/your-org/airi.git
cd airi
pip install -r requirements.txt
```

### 2. Compile Go utilities

```bash
cd go-utils
./build.sh       # or: go build -o <BinaryName> ./<BinaryName>/
cd ..
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Agent system prompt
AGENT_SYSTEM_MESSAGE="You are Airi, a local AI assistant..."

# Google OAuth (for Gmail, Calendar, Drive, Tasks)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_PROJECT_ID=...
GOOGLE_REDIRECT_URI=http://localhost

# Telegram bot (optional)
TELEGRAM_BOT_TOKEN=...

# Qdrant (for RAG)
QDRANT_URL=http://localhost:6333
```

### 4. Start the backend

```bash
python server.py
```

### 5. Launch the TUI

```bash
cd frontend
go run main.go
```

The TUI connects to `ws://localhost:8000/ws/chat` by default. You can also query the REST endpoint directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my disk usage?", "session_id": "my-session"}'
```

---

## Tool Ecosystem

Airi ships with 60+ tools across functional domains. The agent selects tools based on natural language context alone — no explicit invocation syntax required.

### System & Process Management
Battery status, disk space, uptime, CPU load averages, thermals, running processes (sorted by CPU usage), active network connections — all sourced from `/proc`, `/sys`, and kernel syscalls. Includes process termination (graceful and force-kill), and system shutdown/restart/sleep with deliberate delays so the agent can acknowledge before the machine goes dark.

### File System
Full-tree file search with configurable timeout (skipping `/proc`, `/sys`, and hidden directories). File creation and targeted in-place text modification without full rewrites. All agent-created files are scoped to `Airi_created_files/`.

### Shell Execution
Async shell runner with a 120-second timeout and clean stdout/stderr separation. The universal escape hatch — if no dedicated tool covers a task, it goes here.

### Browser Automation
CDP-based browser control via the `agent-browser` binary (Rust). Supports URL navigation, interactive snapshots (with element references like `@e1`), click, fill, type, scroll, JavaScript evaluation, cookie/localStorage access, tab management, and page state diffing. Commands are architecturally classified as **observation** (blocking) or **action** (fire-and-forget). Session persists across commands via a named profile.

### Mobile Device Control
Full automation of iOS simulators/devices (via XCTest) and Android emulators/devices (via ADB), routed through `agent-device`. Platform is auto-detected on first use — you never specify iOS vs Android explicitly. Supports app launch, UI snapshot, element interaction by ID or semantic label, text input, scroll, hardware gestures, clipboard, app state inspection, and performance metrics. A companion `adb_key_press` tool handles keys `agent-device` doesn't expose: Enter, Search, Back, D-pad, volume, media controls, and more.

### Desktop Automation
`xdotool`, `wmctrl`, `scrot`, `xclip` — window control, keyboard/mouse simulation, screenshots, and clipboard access.

### Google Workspace
Full OAuth2 integration with locally stored, auto-refreshing tokens.
- **Gmail:** Read unread messages, search, send, reply (with correct `In-Reply-To`/`References` threading), create drafts.
- **Calendar:** List upcoming events, create and delete events.
- **Drive:** List, search, upload, download files.
- **Tasks:** List pending tasks, add with due dates, complete, delete.

### Telegram
Send messages to contacts defined in `config.yaml` via a bot token. Contact list is retrievable by the agent at runtime.

### Music Playback
VLC-controlled via the RC socket interface. Play songs/playlists/random tracks, pause, stop, skip, set volume — no UI interaction required.

### Camera
Webcam capture via `fswebcam` or `ffmpeg`. Single photo capture (with optional countdown), background video recording with audio, timelapse sequences, and a full captures manager (list, delete).

### Maps & Navigation
Google Maps search and directions via headless browser — no API key required.

### News
RSS-based retrieval for top headlines, topic-specific feeds, and region-specific sources. Structured feed parsing — no fragile scraping.

### Cron Scheduling
List, add, and delete system cron jobs via a compiled Go binary. Job metadata is persisted locally as JSON alongside the crontab entry.

### System Diagnostics
Concurrent health report: CPU load, RAM, disk, thermals, and network ping — gathered in parallel goroutines, returned as structured JSON with automatic warning flags when thresholds are exceeded.

### RAG Pipeline
Qdrant-backed retrieval-augmented generation. Searched transparently during normal query resolution — the agent doesn't need explicit instruction to use it.

### TUI Games
Terminal games (Chess vs Stockfish, Block Breaker, Alien Shooter, Ping Pong) launched in a new terminal window with automatic emulator detection.

---

## Configuration

### `config.yaml`

```yaml
telegram_contacts:
  - name: Alice
    chat_id: "123456789"
  - name: Bob
    chat_id: "987654321"
```

### Adding RAG documents

Edit the `documents` list in `RagSearch.py`:

```python
documents = [
    {"path": "tmp/my_document.pdf", "metadata": {"subject": "Notes", "batch": 2025}},
]
```

Documents are ingested at startup. Re-ingestion is skipped if the file hasn't changed (MD5 check).

### Model selection

Change the model in `server.py`:

```python
# Local (default)
model=Ollama(id="llama3.2:latest")

# Cloud
from agno.models.anthropic import Claude
model=Claude(id="claude-opus-4-20250514")

from agno.models.openai import OpenAIChat
model=OpenAIChat(id="gpt-4o")
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | agno framework |
| Backend | FastAPI, Python 3.11+, asyncio |
| Frontend | Go, Bubbletea TUI |
| Model Serving | Agnostic — Ollama, vLLM, OpenAI, Anthropic, DeepSeek, Groq, etc. |
| Vector DB | Agnostic — Qdrant (default), PgVector, Pinecone, Milvus, and 15+ more |
| Embeddings | Agnostic — Ollama (default), FastEmbed, OpenAI, Cohere, Voyage AI |
| Go Utilities | Compiled binaries in `go-utils/` |
| Browser Automation | agent-browser (CDP, Rust binary) |
| Mobile Automation | agent-device (XCTest + ADB) |
| Desktop Automation | xdotool, wmctrl, scrot, xclip |
| Music | VLC RC socket interface |
| Voice | speech_recognition, spd-say |
| Communication | WebSocket, REST (FastAPI) |

---

## Design Principles

**Local-first, always.** No data leaves the machine unless a tool explicitly sends it somewhere. No telemetry, no cloud sync, no external dependency for core functionality.

**The agent owns its environment.** Airi doesn't call an abstract "computer use" API. Its tools have the same low-level access a developer has at a terminal.

**Observation and action are architecturally distinct.** Commands that return data block until complete. Commands that trigger side effects fire immediately. This is structural — enforced through how timeouts and output handling work across the browser and device layers, not just a naming convention.

**Go is a tool, not an identity.** The compiled binary pattern is used only where Go has a concrete advantage: concurrency, syscalls, low-level parsing. Simple subprocess delegation stays in Python. The architecture commits to the right tool per job, not to a language.

**Memory and session history are different things.** Session state is ephemeral and wiped on shutdown. Long-term memory persists via a separate storage path. The agent accumulates context about the user over time without dragging stale conversation history into new sessions.

---

## Extending Airi

The entire tool contract is: **a Python function, decorated with `@tool`, appended to `TOOLS`.**

```python
# my_new_tool.py
import requests
from agno.tools import tool

@tool
def get_weather(city: str) -> str:
    """Gets current weather for a city."""
    resp = requests.get(f"https://wttr.in/{city}?format=3")
    return resp.text
```

```python
# server.py
from utils.my_new_tool import get_weather

TOOLS = [
    ...
    get_weather,   # ← agent starts using it immediately
]
```

That's the entire integration path — for REST APIs, CLI binaries, local services, hardware peripherals, IoT devices, or any SaaS SDK. No plugin registry, no manifest, no schema upload. If it can be expressed as a Python function, Airi can use it.

---
