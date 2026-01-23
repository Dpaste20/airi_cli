import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "FetchUrls")


@tool
def fetch_urls(query: str, k: int = 5) -> dict:
    if not os.path.exists(BINARY_PATH):
        return {
            "error": f"Binary not found at {BINARY_PATH}. Please compile the Go utility."
        }

    if not query or not query.strip():
        return {"error": "Query cannot be empty"}

    try:
        cmd = [BINARY_PATH, "-query", query, "-k", str(k)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        return json.loads(result.stdout)

    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from Go utility"}
    except subprocess.CalledProcessError as e:
        return {"error": f"Go utility execution failed: {e}"}
    except Exception as e:
        return {"error": str(e)}
