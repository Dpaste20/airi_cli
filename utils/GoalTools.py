import asyncio
import os
import time
from typing import Optional

import aiofiles
import yaml
from agno.tools import tool

GOALS_FILE = os.path.join(os.getcwd(), "UserProfile", "goal.yaml")

GOAL_STATUSES = ("active", "paused", "completed", "abandoned")
SUB_GOAL_STATUSES = ("pending", "active", "completed", "abandoned")
OBJECTIVE_STATUSES = ("pending", "completed")
PRIORITIES = ("low", "medium", "high")

_lock = asyncio.Lock()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _default_state() -> dict:
    return {"goals": [], "next_goal_id": 1}


async def _read_state() -> dict:
    """Loads goal.yaml. Returns a fresh default state if missing/corrupt."""
    if not os.path.exists(GOALS_FILE):
        return _default_state()

    try:
        async with aiofiles.open(GOALS_FILE, "r", encoding="utf-8") as f:
            content = await f.read()
        if not content.strip():
            return _default_state()

        state = yaml.safe_load(content)
        if not isinstance(state, dict):
            return _default_state()

        state.setdefault("goals", [])
        state.setdefault("next_goal_id", 1)
        return state
    except (yaml.YAMLError, IOError) as e:
        print(f"Warning: Could not parse {GOALS_FILE} ({e}). Starting fresh.")
        return _default_state()


async def _write_state(state: dict) -> None:
    """Persists state back to goal.yaml."""
    os.makedirs(os.path.dirname(GOALS_FILE), exist_ok=True)
    dump = yaml.safe_dump(
        state, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    async with aiofiles.open(GOALS_FILE, "w", encoding="utf-8") as f:
        await f.write(dump)


def _find_goal(state: dict, goal_id: int) -> Optional[dict]:
    return next((g for g in state.get("goals", []) if g["id"] == goal_id), None)


def _find_sub_goal(goal: dict, sub_goal_id: int) -> Optional[dict]:
    return next(
        (sg for sg in goal.get("sub_goals", []) if sg["id"] == sub_goal_id), None
    )


def _find_objective(sub_goal: dict, objective_id: int) -> Optional[dict]:
    return next(
        (o for o in sub_goal.get("objectives", []) if o["id"] == objective_id), None
    )


def _next_local_id(items: list) -> int:
    ids = [item["id"] for item in items]
    return max(ids) + 1 if ids else 1


def _touch(obj: dict) -> None:
    obj["updated_at"] = _timestamp()


def _append_log(goal: dict, entry: str) -> None:
    goal.setdefault("logs", []).append({"timestamp": _timestamp(), "entry": entry})


def _status_icon(status: str) -> str:
    return {
        "active": "▶",
        "pending": "○",
        "paused": "⏸",
        "completed": "✓",
        "abandoned": "✗",
    }.get(status, "?")


def _progress(goal: dict) -> str:
    sub_goals = goal.get("sub_goals", [])
    sg_done = sum(1 for sg in sub_goals if sg["status"] == "completed")

    all_objectives = [o for sg in sub_goals for o in sg.get("objectives", [])]
    o_done = sum(1 for o in all_objectives if o["status"] == "completed")

    return (
        f"{sg_done}/{len(sub_goals)} sub-goals · "
        f"{o_done}/{len(all_objectives)} objectives"
    )


@tool
async def create_goal(
    title: str,
    description: str = "",
    priority: str = "medium",
    target_date: str = "",
) -> str:
    """
    Creates a new top-level goal. Multiple goals can be active at once —
    this does NOT replace or clear any existing goal.

    Args:
        title: Short name for the goal (e.g. "Launch v2 of Airi").
        description: Optional longer description of what success looks like.
        priority: One of "low", "medium", "high". Defaults to "medium".
        target_date: Optional free-form target date (e.g. "2026-08-01").

    Returns:
        Confirmation message containing the new goal's ID.
    """
    title = title.strip()
    if not title:
        return "Error: title cannot be empty."

    priority = priority.lower().strip()
    if priority not in PRIORITIES:
        return f"Error: priority must be one of {PRIORITIES}."

    async with _lock:
        state = await _read_state()

        goal_id = state["next_goal_id"]
        now = _timestamp()
        goal = {
            "id": goal_id,
            "title": title,
            "description": description.strip(),
            "priority": priority,
            "status": "active",
            "target_date": target_date.strip() or None,
            "created_at": now,
            "updated_at": now,
            "sub_goals": [],
            "logs": [{"timestamp": now, "entry": f'Goal created: "{title}"'}],
        }
        state["goals"].append(goal)
        state["next_goal_id"] = goal_id + 1

        await _write_state(state)

    return f'✓ Goal #{goal_id} created: "{title}" (priority: {priority})'


@tool
async def list_goals(status_filter: str = "all") -> str:
    """
    Lists all goals with a quick progress summary. Use this to get an
    overview before drilling into a specific goal with get_goal().

    Args:
        status_filter: "all", or one of "active", "paused", "completed",
                       "abandoned" to narrow the list.

    Returns:
        A formatted summary of matching goals.
    """
    state = await _read_state()
    goals = state.get("goals", [])

    status_filter = status_filter.lower().strip()
    if status_filter != "all":
        if status_filter not in GOAL_STATUSES:
            return f"Error: status_filter must be 'all' or one of {GOAL_STATUSES}."
        goals = [g for g in goals if g["status"] == status_filter]

    if not goals:
        return "📋 No goals found. Use create_goal() to start one."

    lines = [f"📋 Goals ({len(goals)}, filter: {status_filter}):\n"]
    for g in goals:
        lines.append(
            f"#{g['id']} {_status_icon(g['status'])} [{g['status'].upper()}] "
            f"({g['priority']}) {g['title']} — {_progress(g)}"
        )
    return "\n".join(lines)


@tool
async def get_goal(goal_id: int, include_log: bool = True, log_limit: int = 10) -> str:
    """
    Retrieves full detail for one goal: description, every sub-goal and
    its objectives, progress counts, and recent activity log.

    Args:
        goal_id: ID of the goal (see list_goals()).
        include_log: Whether to include the recent progress log.
        log_limit: How many recent log entries to show.

    Returns:
        A formatted breakdown of the goal, or an error if not found.
    """
    state = await _read_state()
    goal = _find_goal(state, goal_id)
    if not goal:
        return f"Error: Goal #{goal_id} not found."

    lines = [
        f"🎯 Goal #{goal['id']}: {goal['title']} "
        f"[{goal['status'].upper()}] (priority: {goal['priority']})"
    ]
    if goal.get("description"):
        lines.append(goal["description"])
    if goal.get("target_date"):
        lines.append(f"Target date: {goal['target_date']}")
    lines.append(f"Progress: {_progress(goal)}\n")

    sub_goals = goal.get("sub_goals", [])
    if sub_goals:
        lines.append("📌 Sub-Goals:")
        for sg in sub_goals:
            objectives = sg.get("objectives", [])
            o_done = sum(1 for o in objectives if o["status"] == "completed")
            lines.append(
                f"  [{_status_icon(sg['status'])}] SG{sg['id']}: {sg['title']} "
                f"({sg['status']}) — {o_done}/{len(objectives)} objectives"
            )
            for o in objectives:
                box = "[x]" if o["status"] == "completed" else "[ ]"
                lines.append(f"      {box} O{o['id']}: {o['description']}")
    else:
        lines.append("📌 Sub-Goals: None defined yet.")

    if include_log and goal.get("logs"):
        recent = goal["logs"][-log_limit:]
        lines.append("\n📝 Recent Progress:")
        for log in recent:
            lines.append(f"- {log['timestamp']} — {log['entry']}")

    return "\n".join(lines)


@tool
async def edit_goal(
    goal_id: int,
    title: str = "",
    description: str = "",
    priority: str = "",
    target_date: str = "",
) -> str:
    """
    Edits a goal's metadata in place (title, description, priority, or
    target date). Leave any argument blank to keep its current value.
    This does not touch sub-goals, objectives, or status — use
    update_goal_status() for status changes.

    Args:
        goal_id: ID of the goal to edit.
        title: New title, or blank to keep current.
        description: New description, or blank to keep current.
        priority: New priority ("low"/"medium"/"high"), or blank to keep current.
        target_date: New target date, or blank to keep current.

    Returns:
        Confirmation message, or an error if the goal isn't found.
    """
    if priority and priority.lower().strip() not in PRIORITIES:
        return f"Error: priority must be one of {PRIORITIES}."

    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        changes = []
        if title.strip():
            changes.append(f'title → "{title.strip()}"')
            goal["title"] = title.strip()
        if description.strip():
            changes.append("description updated")
            goal["description"] = description.strip()
        if priority.strip():
            changes.append(f"priority → {priority.lower().strip()}")
            goal["priority"] = priority.lower().strip()
        if target_date.strip():
            changes.append(f"target_date → {target_date.strip()}")
            goal["target_date"] = target_date.strip()

        if not changes:
            return "No changes provided."

        _touch(goal)
        _append_log(goal, "Edited: " + "; ".join(changes))
        await _write_state(state)

    return f"✓ Goal #{goal_id} updated ({'; '.join(changes)})."


@tool
async def update_goal_status(goal_id: int, status: str, note: str = "") -> str:
    """
    Changes a goal's overall status — use this to mark a goal active again,
    paused, completed, or abandoned.

    Args:
        goal_id: ID of the goal.
        status: One of "active", "paused", "completed", "abandoned".
        note: Optional reason, recorded in the activity log.

    Returns:
        Confirmation message, or an error if the goal/status is invalid.
    """
    status = status.lower().strip()
    if status not in GOAL_STATUSES:
        return f"Error: status must be one of {GOAL_STATUSES}."

    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        old_status = goal["status"]
        goal["status"] = status
        _touch(goal)

        entry = f"Status changed from {old_status} → {status}"
        if note.strip():
            entry += f" (Note: {note.strip()})"
        _append_log(goal, entry)

        await _write_state(state)

    return f"✓ Goal #{goal_id} status: {old_status} → {status}."


@tool
async def delete_goal(goal_id: int) -> str:
    """
    Permanently deletes a goal and everything under it (sub-goals,
    objectives, logs). This cannot be undone.

    Args:
        goal_id: ID of the goal to delete.

    Returns:
        Confirmation message, or an error if not found.
    """
    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        state["goals"] = [g for g in state["goals"] if g["id"] != goal_id]
        await _write_state(state)

    return f'✓ Goal #{goal_id} ("{goal["title"]}") deleted.'


@tool
async def add_sub_goal(goal_id: int, title: str) -> str:
    """
    Adds a sub-goal under a main goal — a meaningful chunk of work that
    will later be broken into objectives with add_objective().

    Args:
        goal_id: ID of the parent goal.
        title: Short name for the sub-goal.

    Returns:
        Confirmation with the new sub-goal's ID, or an error if the goal
        isn't found.
    """
    title = title.strip()
    if not title:
        return "Error: title cannot be empty."

    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        sub_goal_id = _next_local_id(goal.get("sub_goals", []))
        now = _timestamp()
        goal.setdefault("sub_goals", []).append(
            {
                "id": sub_goal_id,
                "title": title,
                "status": "pending",
                "created_at": now,
                "updated_at": now,
                "objectives": [],
            }
        )
        _touch(goal)
        _append_log(goal, f"Added sub-goal SG{sub_goal_id}: {title}")

        await _write_state(state)

    return f"✓ Sub-goal SG{sub_goal_id} added to Goal #{goal_id}: {title}"


@tool
async def update_sub_goal_status(goal_id: int, sub_goal_id: int, status: str) -> str:
    """
    Updates a sub-goal's status.

    Args:
        goal_id: ID of the parent goal.
        sub_goal_id: ID of the sub-goal (scoped to this goal).
        status: One of "pending", "active", "completed", "abandoned".

    Returns:
        Confirmation message, or an error if not found / invalid status.
    """
    status = status.lower().strip()
    if status not in SUB_GOAL_STATUSES:
        return f"Error: status must be one of {SUB_GOAL_STATUSES}."

    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        sub_goal = _find_sub_goal(goal, sub_goal_id)
        if not sub_goal:
            return f"Error: Sub-goal SG{sub_goal_id} not found under Goal #{goal_id}."

        old_status = sub_goal["status"]
        sub_goal["status"] = status
        _touch(sub_goal)
        _touch(goal)
        _append_log(
            goal,
            f"Sub-goal SG{sub_goal_id} ('{sub_goal['title']}') {old_status} → {status}",
        )

        await _write_state(state)

    return f"✓ Sub-goal SG{sub_goal_id}: {old_status} → {status}."


@tool
async def delete_sub_goal(goal_id: int, sub_goal_id: int) -> str:
    """
    Deletes a sub-goal and all of its objectives.

    Args:
        goal_id: ID of the parent goal.
        sub_goal_id: ID of the sub-goal to delete.

    Returns:
        Confirmation message, or an error if not found.
    """
    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        sub_goal = _find_sub_goal(goal, sub_goal_id)
        if not sub_goal:
            return f"Error: Sub-goal SG{sub_goal_id} not found under Goal #{goal_id}."

        goal["sub_goals"] = [sg for sg in goal["sub_goals"] if sg["id"] != sub_goal_id]
        _touch(goal)
        _append_log(goal, f"Deleted sub-goal SG{sub_goal_id} ('{sub_goal['title']}')")

        await _write_state(state)

    return f"✓ Sub-goal SG{sub_goal_id} deleted from Goal #{goal_id}."


@tool
async def add_objective(goal_id: int, sub_goal_id: int, description: str) -> str:
    """
    Adds an objective — the smallest trackable unit of work — under a
    sub-goal.

    Args:
        goal_id: ID of the parent goal.
        sub_goal_id: ID of the parent sub-goal.
        description: What needs to be done.

    Returns:
        Confirmation with the new objective's ID, or an error if not found.
    """
    description = description.strip()
    if not description:
        return "Error: description cannot be empty."

    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        sub_goal = _find_sub_goal(goal, sub_goal_id)
        if not sub_goal:
            return f"Error: Sub-goal SG{sub_goal_id} not found under Goal #{goal_id}."

        objective_id = _next_local_id(sub_goal.get("objectives", []))
        now = _timestamp()
        sub_goal.setdefault("objectives", []).append(
            {
                "id": objective_id,
                "description": description,
                "status": "pending",
                "created_at": now,
                "updated_at": now,
            }
        )
        _touch(sub_goal)
        _touch(goal)
        _append_log(
            goal, f"Added objective O{objective_id} to SG{sub_goal_id}: {description}"
        )

        await _write_state(state)

    return (
        f"✓ Objective O{objective_id} added to Sub-goal SG{sub_goal_id} "
        f"(Goal #{goal_id}): {description}"
    )


@tool
async def update_objective_status(
    goal_id: int, sub_goal_id: int, objective_id: int, status: str
) -> str:
    """
    Updates an objective's status. If this completes every objective
    under its sub-goal, the sub-goal is left as-is — call
    update_sub_goal_status() explicitly to mark the sub-goal complete.

    Args:
        goal_id: ID of the parent goal.
        sub_goal_id: ID of the parent sub-goal.
        objective_id: ID of the objective (scoped to this sub-goal).
        status: One of "pending", "completed".

    Returns:
        Confirmation message, or an error if not found / invalid status.
    """
    status = status.lower().strip()
    if status not in OBJECTIVE_STATUSES:
        return f"Error: status must be one of {OBJECTIVE_STATUSES}."

    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        sub_goal = _find_sub_goal(goal, sub_goal_id)
        if not sub_goal:
            return f"Error: Sub-goal SG{sub_goal_id} not found under Goal #{goal_id}."

        objective = _find_objective(sub_goal, objective_id)
        if not objective:
            return f"Error: Objective O{objective_id} not found under SG{sub_goal_id}."

        old_status = objective["status"]
        objective["status"] = status
        _touch(objective)
        _touch(sub_goal)
        _touch(goal)

        objectives = sub_goal.get("objectives", [])
        all_done = bool(objectives) and all(
            o["status"] == "completed" for o in objectives
        )

        _append_log(
            goal,
            f"Objective O{objective_id} ('{objective['description']}') "
            f"{old_status} → {status}",
        )

        await _write_state(state)

    suffix = ""
    if status == "completed" and all_done:
        suffix = (
            f" 🎉 All objectives under SG{sub_goal_id} are now complete — "
            f"consider update_sub_goal_status(goal_id={goal_id}, "
            f"sub_goal_id={sub_goal_id}, status='completed')."
        )
    return f"✓ Objective O{objective_id}: {old_status} → {status}.{suffix}"


@tool
async def delete_objective(goal_id: int, sub_goal_id: int, objective_id: int) -> str:
    """
    Deletes a single objective.

    Args:
        goal_id: ID of the parent goal.
        sub_goal_id: ID of the parent sub-goal.
        objective_id: ID of the objective to delete.

    Returns:
        Confirmation message, or an error if not found.
    """
    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        sub_goal = _find_sub_goal(goal, sub_goal_id)
        if not sub_goal:
            return f"Error: Sub-goal SG{sub_goal_id} not found under Goal #{goal_id}."

        objective = _find_objective(sub_goal, objective_id)
        if not objective:
            return f"Error: Objective O{objective_id} not found under SG{sub_goal_id}."

        sub_goal["objectives"] = [
            o for o in sub_goal["objectives"] if o["id"] != objective_id
        ]
        _touch(sub_goal)
        _touch(goal)
        _append_log(
            goal,
            f"Deleted objective O{objective_id} ('{objective['description']}') "
            f"from SG{sub_goal_id}",
        )

        await _write_state(state)

    return f"✓ Objective O{objective_id} deleted from Sub-goal SG{sub_goal_id}."



@tool
async def log_goal_progress(
    goal_id: int, note: str, sub_goal_id: Optional[int] = None
) -> str:
    """
    Appends a free-form, timestamped progress note to a goal's activity
    log. Optionally tag it to a specific sub-goal for context.

    Args:
        goal_id: ID of the goal to log against.
        note: The progress note text.
        sub_goal_id: Optional sub-goal ID to tag the note with.

    Returns:
        Confirmation message, or an error if the goal/sub-goal isn't found.
    """
    note = note.strip()
    if not note:
        return "Error: note cannot be empty."

    async with _lock:
        state = await _read_state()
        goal = _find_goal(state, goal_id)
        if not goal:
            return f"Error: Goal #{goal_id} not found."

        prefix = ""
        if sub_goal_id is not None:
            sub_goal = _find_sub_goal(goal, sub_goal_id)
            if not sub_goal:
                return (
                    f"Error: Sub-goal SG{sub_goal_id} not found under Goal #{goal_id}."
                )
            prefix = f"[SG{sub_goal_id}] "

        _append_log(goal, f"{prefix}{note}")
        _touch(goal)
        await _write_state(state)

    return f"✓ Progress logged on Goal #{goal_id}: {note}"
