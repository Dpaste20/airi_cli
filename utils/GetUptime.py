import os
import time

from agno.tools import tool


@tool
def get_uptime() -> dict:
    uptime_file = "/proc/uptime"

    if not os.path.exists(uptime_file):
        return {"error": "Uptime information not available"}

    try:
        with open(uptime_file, "r") as f:
            uptime_seconds = float(f.read().split()[0])

        # Convert to hours, minutes, seconds
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)

        return {
            "uptime_seconds": int(uptime_seconds),
            "uptime_readable": f"{days}d {hours}h {minutes}m {seconds}s",
        }

    except Exception as e:
        return {"error": str(e)}
