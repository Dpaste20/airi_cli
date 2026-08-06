import base64
import os

import anydoc
from agno.tools import tool

SUPPORTED_FORMATS = {
    "doc": "Word",
    "docx": "Word",
    "docm": "Word",
    "ppt": "PowerPoint",
    "pps": "PowerPoint",
    "pot": "PowerPoint",
    "pptx": "PowerPoint",
    "pptm": "PowerPoint",
    "ppsx": "PowerPoint",
    "ppsm": "PowerPoint",
    "xls": "Excel",
    "xlsx": "Excel",
    "xlsm": "Excel",
    "xlsb": "Excel",
    "odt": "OpenDocument",
    "ods": "OpenDocument",
    "odp": "OpenDocument",
    "rtf": "Rich Text",
    "epub": "EPUB",
    "csv": "CSV",
    "pdf": "PDF",
}

_ERR_HINTS = {
    "UnsupportedError": "format not supported or scanned/image-only document (no OCR)",
    "MalformedError": "structurally unusable, no meaningful content could be extracted",
    "EncryptedError": "encrypted or password-protected",
    "ResourceLimitError": "exceeded a safety limit (decompression, nesting, node count)",
    "MissingPartError": "a required part is missing",
}


def _error_block(name: str, exc: Exception) -> str:
    exc_name = type(exc).__name__
    hint = _ERR_HINTS.get(exc_name, str(exc))
    return f"[File '{name}': conversion failed — {hint}]"


def convert_bytes_to_markdown(raw: bytes, name: str, mime_type: str = "") -> str:
    """Convert a document's raw bytes into Markdown via anydoc."""
    fmt = anydoc.format_from_bytes(raw)
    if fmt is None:
        ext = os.path.splitext(name)[1].lstrip(".").lower()
        fmt = anydoc.format_from_extension(f".{ext}") if ext else None

    if fmt is None:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")

    try:
        return anydoc.to_markdown_bytes(raw, fmt)
    except anydoc.ConvertError as exc:
        return _error_block(name, exc)
    except ValueError as exc:
        return _error_block(name, exc)
    except Exception as exc:
        return f"[File '{name}': conversion error — {exc}]"


def decode_file_content(name: str, content: str) -> str:
    """Decode a base64 attachment body into Markdown/text for the prompt."""
    try:
        raw = base64.b64decode(content)
    except Exception as exc:
        return f"[File '{name}': could not decode base64 — {exc}]"
    return convert_bytes_to_markdown(raw, name)


@tool
def convert_document(path: str) -> str:
    """Converts an office document (Word, PowerPoint, Excel, OpenDocument, RTF,
    EPUB, CSV, PDF) at the given path into clean Markdown. Use this to read any
    document file the user mentions, e.g. before summarising it or adding it to
    the knowledge base."""
    if not path or not path.strip():
        return "Error: path cannot be empty"

    path = os.path.expanduser(path.strip())
    if not os.path.isfile(path):
        return f"Error: file not found: {path}"

    try:
        return anydoc.to_markdown(path)
    except anydoc.ConvertError as exc:
        return _error_block(os.path.basename(path), exc)
    except OSError as exc:
        return f"[File '{os.path.basename(path)}': could not be read — {exc}]"
    except Exception as exc:
        return f"[File '{os.path.basename(path)}': conversion error — {exc}]"
