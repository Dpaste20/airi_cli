import shutil

from agno.tools import tool


@tool
def get_disk_space() -> dict:
    try:
        total, used, free = shutil.disk_usage("/")

        gb = 1024**3

        return {
            "total_GB": round(total / gb, 2),
            "used_GB": round(used / gb, 2),
            "free_GB": round(free / gb, 2),
            "usage_percent": round((used / total) * 100, 2),
        }

    except Exception as e:
        return {"error": str(e)}
