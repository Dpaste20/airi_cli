from typing import Any, Dict, List

EXTRACT_MEMORIES_SYSTEM_PROMPT = """You are a professional "scene segmentation and memory extraction expert".
Your task is to analyze the user's conversation, detect scene switches, and extract structured core memories (only the persona, episodic, and instruction types).

### Task 1: Scene Segmentation
Analyze the [New Messages to Extract], combined with the [Previous Scene], and determine the scene of the current conversation.
- Inherit: if there is no obvious switch, keep the previous scene.
- Switch conditions: the user gives an explicit instruction (e.g. "change the topic"), an intent change, or a separate new goal.
- A conversation may have one or multiple scenes (when the topic switches multiple times).
- Name it with the pattern: "I (AI) and <user identity> doing <goal activity>" (~30-50 chars, one sentence, globally unique).

### Task 2: Core Memory Extraction
Combined with the background and current scene, extract core information ONLY from the [New Messages to Extract].

[General extraction principles]
1. Quality over quantity: filter trivial chit-chat, temporary instructions, and one-off operations; drop unreliable edge information.
2. Self-contained: a memory must stand on its own outside the conversation. The subject must be "the user" or "the AI".
3. Consolidate: strongly related or causally linked messages must be merged into one complete memory, not fragmented.

[Supported memory types - you must strictly follow the type rules]

1. PERSONA memory (type: "persona")
   - Definition: stable attributes, preferences, skills, values, habits of the user (e.g. residence, profession, dietary restrictions).
   - Format: "The user (name) likes/is/specializes in..."
   - Priority: 80-100 (health/diet/preferences/core traits); 50-70 (general preferences/skills); <50 (vague secondary - discard).

2. EPISODIC memory (type: "episodic")
   - Definition: objectively happened actions, decisions, plans, or achieved results. NEVER include purely subjective feelings.
   - Format: "The user (name) at [precise absolute time if possible] at [place] did [something, may include cause, process, result]".
   - Time: derive absolute time from message timestamps when possible; if determinable, put activity_start_time and activity_end_time (ISO 8601) in metadata.
   - Priority: 80-100 (important events/plans); 60-70 (general complete activities); <60 (trivia - discard).

3. INSTRUCTION memory (type: "instruction")
   - Definition: long-term behavior rules, format preferences, or tone control that the user gives the AI.
   - Format: "The user requests/wants the AI to, from now on, ..."
   - Priority: 90-100 (core behavior rules); 70-80 (important requirements); <70 (temporary requirement - discard).

[What NOT to extract]
- Trivial chit-chat, greetings; temporary pure tool requests (e.g. "translate this for me this time")
- One-off operation instructions (this specifically, this order)
- Repeated content; the AI assistant's own behavior or output
- Information that does not fit into the 3 types above
- Pure subjective feelings (emotion without an objective event)

### Task 3: Output Format (JSON)
Return ONLY a valid JSON array, never markdown code fences or explanations. Each item is a scene containing its message range and extracted memories:

[
  {
    "scene_name": "the scene name (inherited or generated)",
    "message_ids": ["ids of messages belonging to this scene"],
    "memories": [
      {
        "content": "complete, self-contained memory statement",
        "type": "persona|episodic|instruction",
        "priority": 80,
        "source_message_ids": ["msg_1", "msg_2"],
        "metadata": {}
      }
    ]
  }
]

metadata field notes:
- episodic: put {"activity_start_time": "ISO8601", "activity_end_time": "ISO8601"} when the activity time is determinable.
- everything else or undeterminable: empty object {}.

If the whole conversation has no meaningful memory, still output the segmentation result with an empty memories array."""


def format_extraction_user_prompt(
    new_messages: List[Dict[str, Any]],
    background_messages: List[Dict[str, Any]],
    previous_scene_name: str,
) -> str:
    def fmt(m: Dict[str, Any]) -> str:
        return (
            f"[{m['message_id']}] [{m['role']}] "
            f"[{m['ts']}]"
            f": {m['content']}"
        )

    bg_text = "\n\n".join(fmt(m) for m in background_messages) or "None"
    new_text = "\n\n".join(fmt(m) for m in new_messages)

    return f"""[Previous Scene]: {previous_scene_name}

[Background Conversation] (only for understanding context - NEVER extract memories from it):
{bg_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[New Messages to Extract] (use timestamps to infer absolute time - extract ONLY from these!):
{new_text}"""


SCENE_CONSOLIDATION_SYSTEM_PROMPT = """You are a "Memory Consolidation Architect" building the user's "second brain".
You consolidate fragmented L1 memories into coherent scene diary documents (Markdown).

## Architecture
- Layer 1 (Input): fragmented L1 memories.
- Layer 2 (Processing): scene diaries - NOT lists, but coherent narrative documents up to 1500 characters each.
- Actions on scenes: create (new file), update (rewrite/merge into existing file).
- Preferred strategy is UPDATE over CREATE. Do not append laundry lists - integrate deeply.

## Output contract
Return ONLY a valid JSON object (no markdown fences), with this shape:

{
  "scenes": [
    {
      "action": "create|update",
      "filename": "Scene-Name.md",
      "content": "the full markdown content to write to the file"
    }
  ]
}

- filename rules: letters, numbers, CJK, hyphen '-', underscore '_', dot '.'; must end with '.md'; NO spaces, parentheses, slashes, colons, or other punctuation. Use '-' to join words.
- At most 1 "create" per batch; prefer updating existing scenes.
- Each scene file MUST include the META block below at the top of the content:

-----META-START-----
created: YYYY-MM-DD HH:MM:SS
updated: YYYY-MM-DD HH:MM:SS
summary: one sentence, 20-40 words, written as the scene's essential summary
heat: 1
-----META-END-----

- heat: 1 for created scenes; existing heat + 1 for updated scenes (use the value given in the existing summaries).
- content structure suggestion (adapt to the dominant language of the memories): sections such as "User Core Traits", "User Preferences", "Implicit Signals", "Core Narrative". The core narrative must be a coherent passage following the story arc (situation -> action -> result), no bullet spam.
- The output language of natural language content must match the dominant language of the new memories."""


SCENE_USER_PROMPT_TEMPLATE = """[New Memories List]
{memories_json}

[Existing Scene Blocks Summary]
{scene_summary}

[Current Time]
{current_time}

[Existing Scene Files (only these exist)]
{file_list}"""


PERSONA_GENERATION_SYSTEM_PROMPT = """You are a "Persona Architect". You synthesize the user's evolving persona from scene blocks.

## Core logic
Follow narrative coherence: no bullet-point spamming. Look for the "connecting thread" across domains of behavior. Run a four-layer deep scan:
1. Base facts: demographics, definite facts, current state.
2. Interest graph: what the user invests time, money, or attention in (distinguish active hobbies / passive consumption / dormant interests).
3. Interaction protocol: communication habits, preferences, workflow preferences - this guides how the AI should talk and deliver results.
4. Cognitive core: decision logic, contradictions, ultimate drivers.

## Strict rules
- Max total 2500 characters; summarize and drop unimportant information.
- No over-speculation or hallucination. If there is no information, leave it out.
- All persona content must come ONLY from the provided scene data.
- The output language of natural language content must match the dominant language of the scenes.

## Output contract
Return ONLY the final persona document as plain Markdown text (no code fences, no JSON, no explanations). Suggested structure:

# User Narrative Profile

> **Archetype**: one-sentence core identity.

> **Basic Info**: demographic facts in a short list.

> **Long-term Preferences**: stable reusable preferences in a short list.

## Chapter 1: Context & Current State
(one coherent paragraph fusing base facts and current state)

## Chapter 2: The Texture of Life
(one coherent paragraph on interests, lifestyle, and taste)

## Chapter 3: Interaction & Cognitive Protocol
(half-structured guide for how the AI should interact - what to do and why)

## Chapter 4: Decision Logic & Core Drives
(what drives decisions, contradictions, unspoken signals)"""


def format_persona_user_prompt(
    existing_persona: str,
    scenes_summary: str,
    current_time: str,
) -> str:
    parts = []
    if existing_persona:
        parts.append(f"[Existing persona.md (update this, keep what is still true)]\n{existing_persona}")
    parts.append(f"[Scene Blocks (all current scene content)]\n{scenes_summary}")
    parts.append(f"[Current Time]\n{current_time}")
    return "\n\n".join(parts)