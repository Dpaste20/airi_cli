import json
import os
from typing import Any, Dict, List, Optional

import requests
from agno.tools import tool

NOTION_VERSION = "2022-06-28"
BASE_URL = "https://api.notion.com/v1"


def _headers() -> Dict[str, str]:
    token = os.getenv("NOTION_API_KEY")
    if not token:
        raise ValueError("NOTION_API_KEY not set in environment.")
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _get(path: str, params: Optional[Dict] = None) -> Dict:
    resp = requests.get(
        f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: Dict) -> Dict:
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _patch(path: str, body: Dict) -> Dict:
    resp = requests.patch(
        f"{BASE_URL}{path}", headers=_headers(), json=body, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


def _rich_text(text: str) -> List[Dict]:
    """Wrap a plain string into a Notion rich_text array."""
    return [{"type": "text", "text": {"content": text}}]


def _paragraph_block(text: str) -> Dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": _rich_text(text)},
    }


def _heading_block(text: str, level: int = 2) -> Dict:
    h = f"heading_{max(1, min(3, level))}"
    return {"object": "block", "type": h, h: {"rich_text": _rich_text(text)}}


def _bullet_block(text: str) -> Dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": _rich_text(text)},
    }


def _todo_block(text: str, checked: bool = False) -> Dict:
    return {
        "object": "block",
        "type": "to_do",
        "to_do": {"rich_text": _rich_text(text), "checked": checked},
    }


def _extract_plain_text(rich_text: List[Dict]) -> str:
    return "".join(rt.get("plain_text", "") for rt in rich_text)


def _summarise_block(block: Dict) -> str:
    """Return a readable one-liner for a single block."""
    btype = block.get("type", "unknown")
    data = block.get(btype, {})
    rt = data.get("rich_text", [])
    text = _extract_plain_text(rt)

    if btype == "to_do":
        tick = "✅" if data.get("checked") else "☐"
        return f"  {tick} {text}"
    if btype in ("bulleted_list_item", "numbered_list_item"):
        return f"  • {text}"
    if btype.startswith("heading_"):
        level = btype[-1]
        return f"\n{'#' * int(level)} {text}"
    if btype == "divider":
        return "  ─────────────────────"
    if btype == "code":
        lang = data.get("language", "")
        return f"  ```{lang}\n  {text}\n  ```"
    if btype == "image":
        url = (data.get("external") or data.get("file") or {}).get("url", "")
        return f"  [Image] {url}"
    if btype == "child_page":
        return f"  📄 Child page: {data.get('title', '')}"
    if btype == "child_database":
        return f"  🗃  Child database: {data.get('title', '')}"
    return f"  {text}" if text else f"  [{btype}]"


@tool
def search_notion(query: str, filter_type: str = "all") -> str:
    """
    Search across all pages and databases the integration can access.

    Args:
        query:       Search text. Pass an empty string to list everything.
        filter_type: 'page', 'database', or 'all' (default).

    Returns:
        Formatted list of matching objects with their IDs, titles, and URLs.
    """
    body: Dict[str, Any] = {"query": query, "page_size": 20}

    if filter_type in ("page", "database"):
        body["filter"] = {"value": filter_type, "property": "object"}

    try:
        data = _post("/search", body)
    except Exception as e:
        return f"Error searching Notion: {e}"

    results = data.get("results", [])
    if not results:
        return f"No Notion objects found for '{query}'."

    lines = [f"Found {len(results)} result(s) for '{query}':\n"]
    for obj in results:
        otype = obj.get("object", "?")
        oid = obj.get("id", "?")
        url = obj.get("url", "")

        title = "(untitled)"
        if otype == "page":
            props = obj.get("properties", {})
            for prop in props.values():
                if prop.get("type") == "title":
                    title = _extract_plain_text(prop["title"]) or title
                    break
        elif otype == "database":
            title = _extract_plain_text(obj.get("title", [])) or title

        icon_char = "📄" if otype == "page" else "🗃 "
        lines.append(
            f"  {icon_char} [{otype.upper()}] {title}\n"
            f"       ID : {oid}\n"
            f"       URL: {url}"
        )

    return "\n".join(lines)


@tool
def get_page(page_id: str) -> str:
    """
    Retrieves the title, properties, and full block content of a Notion page.

    Args:
        page_id: The Notion page ID (with or without dashes).

    Returns:
        Structured text representation of the page.
    """
    pid = page_id.replace("-", "")
    try:
        page = _get(f"/pages/{pid}")
        blocks = _get(f"/blocks/{pid}/children", {"page_size": 100})
    except Exception as e:
        return f"Error fetching page '{page_id}': {e}"

    title = "(untitled)"
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            title = _extract_plain_text(prop["title"]) or title
            break

    lines = [f"# {title}", f"ID : {page.get('id')}", f"URL: {page.get('url', '')}", ""]

    for name, prop in page.get("properties", {}).items():
        ptype = prop.get("type")
        if ptype == "title":
            continue
        value = ""
        if ptype == "rich_text":
            value = _extract_plain_text(prop.get("rich_text", []))
        elif ptype == "select":
            sel = prop.get("select") or {}
            value = sel.get("name", "")
        elif ptype == "multi_select":
            value = ", ".join(s["name"] for s in prop.get("multi_select", []))
        elif ptype == "date":
            d = prop.get("date") or {}
            value = d.get("start", "")
        elif ptype == "checkbox":
            value = "✅" if prop.get("checkbox") else "☐"
        elif ptype == "number":
            value = str(prop.get("number", ""))
        elif ptype == "url":
            value = prop.get("url", "") or ""
        elif ptype == "email":
            value = prop.get("email", "") or ""
        elif ptype == "phone_number":
            value = prop.get("phone_number", "") or ""
        elif ptype == "people":
            value = ", ".join(p.get("name", "") for p in prop.get("people", []))
        elif ptype == "status":
            st = prop.get("status") or {}
            value = st.get("name", "")
        elif ptype == "relation":
            value = f"{len(prop.get('relation', []))} linked page(s)"
        elif ptype == "formula":
            fval = prop.get("formula", {})
            value = str(fval.get(fval.get("type", ""), ""))

        if value:
            lines.append(f"**{name}**: {value}")

    lines.append("\n--- Content ---")
    for block in blocks.get("results", []):
        lines.append(_summarise_block(block))

    return "\n".join(lines)


@tool
def create_page(
    parent_id: str,
    title: str,
    content: str = "",
    parent_type: str = "page",
) -> str:
    """
    Creates a new Notion page under an existing page or database.

    Args:
        parent_id:   ID of the parent page or database.
        title:       Title of the new page.
        content:     Optional body text (becomes a paragraph block).
        parent_type: 'page' (default) or 'database'.

    Returns:
        ID and URL of the newly created page, or an error message.
    """
    pid = parent_id.replace("-", "")

    if parent_type == "database":
        parent = {"database_id": pid}
        properties = {"Name": {"title": _rich_text(title)}}
    else:
        parent = {"page_id": pid}
        properties = {"title": {"title": _rich_text(title)}}

    body: Dict[str, Any] = {"parent": parent, "properties": properties}

    if content.strip():
        paragraphs = [_paragraph_block(p) for p in content.split("\n") if p.strip()]
        if paragraphs:
            body["children"] = paragraphs

    try:
        page = _post("/pages", body)
    except Exception as e:
        return f"Error creating page: {e}"

    return (
        f"✓ Page created: '{title}'\n"
        f"  ID : {page.get('id')}\n"
        f"  URL: {page.get('url', '')}"
    )


@tool
def update_page_properties(page_id: str, properties_json: str) -> str:
    """
    Updates one or more properties on an existing Notion page.

    Pass properties as a JSON string. Each key is the property name;
    values follow the Notion property value format.

    Common examples:
        Update a title:
            {"Name": {"title": [{"text": {"content": "New Title"}}]}}
        Set a select:
            {"Status": {"select": {"name": "Done"}}}
        Set a checkbox:
            {"Completed": {"checkbox": true}}
        Set a date:
            {"Due": {"date": {"start": "2025-06-01"}}}
        Set a number:
            {"Priority": {"number": 3}}
        Set a URL:
            {"Link": {"url": "https://example.com"}}

    Args:
        page_id:         The Notion page ID.
        properties_json: JSON string of property updates.

    Returns:
        Confirmation with the updated page URL, or an error message.
    """
    pid = page_id.replace("-", "")
    try:
        props = json.loads(properties_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON for properties — {e}"

    try:
        page = _patch(f"/pages/{pid}", {"properties": props})
    except Exception as e:
        return f"Error updating page '{page_id}': {e}"

    return f"✓ Page updated.\n  ID : {page.get('id')}\n  URL: {page.get('url', '')}"


@tool
def append_to_page(page_id: str, blocks_json: str) -> str:
    """
    Appends one or more blocks to the end of a Notion page.

    Pass blocks as a JSON string — an array of Notion block objects,
    or a list of shorthand dicts with keys 'type' and 'text':

        [
          {"type": "heading_2", "text": "Summary"},
          {"type": "paragraph",  "text": "This is the body."},
          {"type": "bulleted_list_item", "text": "First item"},
          {"type": "to_do", "text": "Ship it", "checked": false}
        ]

    For full Notion block format, pass native block objects directly.

    Args:
        page_id:     The Notion page ID.
        blocks_json: JSON array of block objects or shorthand dicts.

    Returns:
        Confirmation or error message.
    """
    pid = page_id.replace("-", "")
    try:
        raw_blocks = json.loads(blocks_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON for blocks — {e}"

    normalised: List[Dict] = []
    for b in raw_blocks:
        if "object" in b:
            normalised.append(b)
            continue
        btype = b.get("type", "paragraph")
        text = b.get("text", "")
        if btype == "to_do":
            normalised.append(_todo_block(text, b.get("checked", False)))
        elif btype == "bulleted_list_item":
            normalised.append(_bullet_block(text))
        elif btype.startswith("heading_"):
            lvl = int(btype.split("_")[-1]) if btype[-1].isdigit() else 2
            normalised.append(_heading_block(text, lvl))
        else:
            normalised.append(_paragraph_block(text))

    if not normalised:
        return "Error: No valid blocks provided."

    try:
        _patch(f"/blocks/{pid}/children", {"children": normalised})
    except Exception as e:
        return f"Error appending blocks to page '{page_id}': {e}"

    return f"✓ {len(normalised)} block(s) appended to page '{page_id}'."


@tool
def query_database(
    database_id: str,
    filter_json: str = "",
    sort_property: str = "",
    sort_direction: str = "ascending",
    page_size: int = 20,
) -> str:
    """
    Queries a Notion database and returns all matching entries.

    Args:
        database_id:    The Notion database ID.
        filter_json:    Optional Notion filter object as a JSON string.
                        Example — entries where Status = "Done":
                        {"property": "Status", "select": {"equals": "Done"}}
        sort_property:  Optional property name to sort by.
        sort_direction: 'ascending' (default) or 'descending'.
        page_size:      Max entries to return (default 20, max 100).

    Returns:
        Formatted list of database entries with their properties.
    """
    did = database_id.replace("-", "")
    body: Dict[str, Any] = {"page_size": min(page_size, 100)}

    if filter_json.strip():
        try:
            body["filter"] = json.loads(filter_json)
        except json.JSONDecodeError as e:
            return f"Error: Invalid filter JSON — {e}"

    if sort_property.strip():
        body["sorts"] = [{"property": sort_property, "direction": sort_direction}]

    try:
        data = _post(f"/databases/{did}/query", body)
    except Exception as e:
        return f"Error querying database '{database_id}': {e}"

    results = data.get("results", [])
    if not results:
        return "No entries found matching the query."

    lines = [f"Found {len(results)} entries:\n"]
    for i, page in enumerate(results, 1):
        pid = page.get("id", "?")
        url = page.get("url", "")
        props = page.get("properties", {})

        prop_parts = []
        for name, prop in props.items():
            ptype = prop.get("type")
            value = ""
            if ptype == "title":
                value = _extract_plain_text(prop.get("title", []))
            elif ptype == "rich_text":
                value = _extract_plain_text(prop.get("rich_text", []))
            elif ptype == "select":
                sel = prop.get("select") or {}
                value = sel.get("name", "")
            elif ptype == "multi_select":
                value = ", ".join(s["name"] for s in prop.get("multi_select", []))
            elif ptype == "date":
                d = prop.get("date") or {}
                value = d.get("start", "")
            elif ptype == "checkbox":
                value = "✅" if prop.get("checkbox") else "☐"
            elif ptype == "number":
                n = prop.get("number")
                value = str(n) if n is not None else ""
            elif ptype == "url":
                value = prop.get("url", "") or ""
            elif ptype == "email":
                value = prop.get("email", "") or ""
            elif ptype == "status":
                st = prop.get("status") or {}
                value = st.get("name", "")
            elif ptype == "people":
                value = ", ".join(p.get("name", "") for p in prop.get("people", []))
            elif ptype == "formula":
                fval = prop.get("formula", {})
                value = str(fval.get(fval.get("type", ""), ""))
            if value:
                prop_parts.append(f"{name}: {value}")

        summary = " | ".join(prop_parts) if prop_parts else "(no properties)"
        lines.append(f"  {i}. {summary}\n     ID : {pid}\n     URL: {url}")

    if data.get("has_more"):
        lines.append("\n  (More results exist — increase page_size or add a filter.)")

    return "\n".join(lines)


@tool
def get_database_schema(database_id: str) -> str:
    """
    Returns the schema (property names and types) of a Notion database.
    Useful before creating entries or building filter queries.

    Args:
        database_id: The Notion database ID.

    Returns:
        Property names, types, and available options for select/status fields.
    """
    did = database_id.replace("-", "")
    try:
        db = _get(f"/databases/{did}")
    except Exception as e:
        return f"Error fetching database schema '{database_id}': {e}"

    title = _extract_plain_text(db.get("title", [])) or "(untitled)"
    lines = [
        f"Database: {title}",
        f"ID : {db.get('id')}",
        f"URL: {db.get('url', '')}",
        "",
        "Properties:",
    ]

    for name, prop in db.get("properties", {}).items():
        ptype = prop.get("type", "?")
        detail = ""

        if ptype == "select":
            opts = [o["name"] for o in prop.get("select", {}).get("options", [])]
            detail = f" → options: {opts}" if opts else ""
        elif ptype == "multi_select":
            opts = [o["name"] for o in prop.get("multi_select", {}).get("options", [])]
            detail = f" → options: {opts}" if opts else ""
        elif ptype == "status":
            opts = [o["name"] for o in prop.get("status", {}).get("options", [])]
            detail = f" → options: {opts}" if opts else ""
        elif ptype == "relation":
            rel = prop.get("relation", {})
            detail = f" → relates to DB: {rel.get('database_id', '?')}"
        elif ptype == "formula":
            expr = prop.get("formula", {}).get("expression", "")
            detail = f" → formula: {expr}" if expr else ""
        elif ptype == "rollup":
            r = prop.get("rollup", {})
            detail = f" → rollup of '{r.get('rollup_property_name', '')}' via '{r.get('relation_property_name', '')}'"

        lines.append(f"  • {name:30s}  [{ptype}]{detail}")

    return "\n".join(lines)


@tool
def create_database_entry(database_id: str, properties_json: str) -> str:
    """
    Creates a new entry (row) in a Notion database.

    Properties must be passed as a JSON string following Notion's format.
    Use get_database_schema first to see available property names and types.

    Common property formats:
        Title:      {"Name": {"title": [{"text": {"content": "Entry Name"}}]}}
        Select:     {"Status": {"select": {"name": "In Progress"}}}
        Multi-sel:  {"Tags": {"multi_select": [{"name": "AI"}, {"name": "Python"}]}}
        Date:       {"Due": {"date": {"start": "2025-06-15"}}}
        Checkbox:   {"Done": {"checkbox": false}}
        Number:     {"Score": {"number": 42}}
        URL:        {"Link": {"url": "https://github.com"}}
        Email:      {"Contact": {"email": "hi@example.com"}}
        Text:       {"Notes": {"rich_text": [{"text": {"content": "Some text"}}]}}

    Args:
        database_id:     The Notion database ID.
        properties_json: JSON string of property values for the new entry.

    Returns:
        ID and URL of the created entry, or an error message.
    """
    did = database_id.replace("-", "")
    try:
        props = json.loads(properties_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON for properties — {e}"

    body = {
        "parent": {"database_id": did},
        "properties": props,
    }

    try:
        page = _post("/pages", body)
    except Exception as e:
        return f"Error creating database entry: {e}"

    return (
        f"✓ Database entry created.\n"
        f"  ID : {page.get('id')}\n"
        f"  URL: {page.get('url', '')}"
    )


@tool
def delete_page(page_id: str) -> str:
    """
    Archives (soft-deletes) a Notion page or database entry.
    Archived pages can be restored from the Notion UI.

    Args:
        page_id: The Notion page ID.

    Returns:
        Confirmation or error message.
    """
    pid = page_id.replace("-", "")
    try:
        _patch(f"/pages/{pid}", {"archived": True})
    except Exception as e:
        return f"Error archiving page '{page_id}': {e}"

    return f"✓ Page '{page_id}' archived successfully."
