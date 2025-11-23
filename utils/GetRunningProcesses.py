import subprocess

from agno.tools import tool


@tool
def get_running_processes(limit: int = 10) -> list:
    """
    Returns a list of the top running processes sorted by CPU usage.

    Args:
        limit (int): The number of processes to return (default is 10).
    """
    try:
        # Run 'ps' command to get process info
        # -e: Select all processes
        # -o: Output format (pid, user, cpu usage, memory usage, command name)
        # --sort=-%cpu: Sort by CPU usage descending
        cmd = ["ps", "-eo", "pid,user,%cpu,%mem,comm", "--sort=-%cpu"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        processes = []
        lines = result.stdout.strip().split("\n")

        # Skip the header (lines[0]) and iterate up to the limit
        # We use lines[1 : limit+1] to get the top N results
        for line in lines[1 : limit + 1]:
            # Split by whitespace, max 4 splits to preserve command name if it has spaces
            parts = line.split(None, 4)

            if len(parts) >= 5:
                processes.append(
                    {
                        "pid": int(parts[0]),
                        "user": parts[1],
                        "cpu": float(parts[2]),
                        "memory": float(parts[3]),
                        "command": parts[4],
                    }
                )

        return processes

    except subprocess.CalledProcessError:
        return [{"error": "Failed to execute ps command"}]
    except Exception as e:
        return [{"error": str(e)}]
