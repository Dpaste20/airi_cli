import subprocess
import time

SESSION_ID = "map_bot"


def map_search(query: str) -> str:
    """
    Searches Google Maps for a specific location or category.
    Runs in a dedicated headless browser session.
    """

    search_url = f"https://www.google.com/maps/search/{query}"

    try:
        subprocess.run(
            ["agent-browser", "--session", SESSION_ID, "open", search_url], check=True
        )

        subprocess.run(
            [
                "agent-browser",
                "--session",
                SESSION_ID,
                "wait",
                "--load",
                "domcontentloaded",
            ],
            check=False,
        )

        time.sleep(3)

        result = subprocess.run(
            ["agent-browser", "--session", SESSION_ID, "get", "text", "body"],
            capture_output=True,
            text=True,
            check=True,
        )

        subprocess.run(["agent-browser", "--session", SESSION_ID, "close"], check=False)

        scraped_text = result.stdout.strip()
        if not scraped_text:
            return "Map Search: Found no readable text."

        return f"--- MAP SEARCH RESULTS FOR '{query}' ---\n{scraped_text[:4000]}..."

    except Exception as e:
        subprocess.run(["agent-browser", "--session", SESSION_ID, "close"], check=False)
        return f"Error executing Map Search: {str(e)}"


def get_directions(start: str, end: str) -> str:
    """
    Gets directions between two points in a headless session.
    """
    directions_url = f"https://www.google.com/maps/dir/{start}/{end}"

    try:
        subprocess.run(
            ["agent-browser", "--session", SESSION_ID, "open", directions_url],
            check=True,
        )

        subprocess.run(
            [
                "agent-browser",
                "--session",
                SESSION_ID,
                "wait",
                "--load",
                "domcontentloaded",
            ],
            check=False,
        )

        time.sleep(3)

        result = subprocess.run(
            ["agent-browser", "--session", SESSION_ID, "get", "text", "body"],
            capture_output=True,
            text=True,
            check=True,
        )

        subprocess.run(["agent-browser", "--session", SESSION_ID, "close"], check=False)

        return (
            f"--- DIRECTIONS FROM {start} TO {end} ---\n{result.stdout.strip()[:4000]}"
        )

    except Exception as e:
        subprocess.run(["agent-browser", "--session", SESSION_ID, "close"], check=False)
        return f"Error getting directions: {str(e)}"
