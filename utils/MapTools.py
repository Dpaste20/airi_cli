import subprocess
import time


def map_search(query: str) -> str:
    """
    Searches Google Maps for a specific location or category.
    """

    search_url = f"https://www.google.com/maps/search/{query}"

    # print(f"Map Tool: Searching for '{query}'...")

    try:
        subprocess.run(["agent-browser", "open", search_url], check=True)

        subprocess.run(
            ["agent-browser", "wait", "--load", "domcontentloaded"], check=False
        )

        time.sleep(3)

        # Scrape
        result = subprocess.run(
            ["agent-browser", "get", "text", "body"],
            capture_output=True,
            text=True,
            check=True,
        )

        scraped_text = result.stdout.strip()
        if not scraped_text:
            return "Map Search: Found no readable text."

        return f"--- MAP SEARCH RESULTS FOR '{query}' ---\n{scraped_text[:4000]}..."

    except Exception as e:
        return f"Error executing Map Search: {str(e)}"


def get_directions(start: str, end: str) -> str:
    """
    Gets directions between two points.
    """
    directions_url = f"https://www.google.com/maps/dir/{start}/{end}"

    # print(f"Map Tool: Directions from '{start}' to '{end}'...")

    try:
        subprocess.run(["agent-browser", "open", directions_url], check=True)

        subprocess.run(
            ["agent-browser", "wait", "--load", "domcontentloaded"], check=False
        )
        time.sleep(3)

        result = subprocess.run(
            ["agent-browser", "get", "text", "body"],
            capture_output=True,
            text=True,
            check=True,
        )

        return (
            f"--- DIRECTIONS FROM {start} TO {end} ---\n{result.stdout.strip()[:4000]}"
        )

    except Exception as e:
        return f"Error getting directions: {str(e)}"
