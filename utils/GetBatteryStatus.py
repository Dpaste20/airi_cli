import os

from agno.tools import tool


@tool
def get_battery_status() -> dict:
    """
    Reads battery information from the Linux sysfs interface.
    Returns a dictionary with percentage and status, or None if not found.
    """
    power_supply_path = "/sys/class/power_supply"

    if not os.path.exists(power_supply_path):
        return {"error": "Battery information not available"}

    for item in os.listdir(power_supply_path):
        if item.startswith("BAT"):
            battery_path = os.path.join(power_supply_path, item)

            try:
                with open(os.path.join(battery_path, "capacity"), "r") as f:
                    capacity = f.read().strip()

                with open(os.path.join(battery_path, "status"), "r") as f:
                    status = f.read().strip()

                return {
                    "battery_id": item,
                    "percentage": int(capacity),
                    "status": status,
                }
            except IOError:
                continue

    return {"error": "No battery found"}
