import os

from agno.tools import tool


@tool
def get_system_logs(lines: int = 100) -> dict:
    log_file = "/var/log/syslog"

    if not os.path.exists(log_file):
        return {"error": "System log not found or inaccessible"}

    try:
        # Read only the last N lines efficiently
        with open(log_file, "r") as f:
            data = f.readlines()[-lines:]

        return {"lines_returned": len(data), "logs": [line.strip() for line in data]}

    except PermissionError:
        return {"error": "Permission denied. Try running with sudo."}
    except Exception as e:
        return {"error": str(e)}
