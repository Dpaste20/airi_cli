import json
import os
import subprocess

from agno.tools import tool

BINARY_PATH = os.path.join(os.getcwd(), "go-utils", "GetIPInfo")


@tool
def get_ip_info() -> dict:
    """
    Retrieves detailed system network information including:
    - Public IP, City, Region, Country, Coordinates
    - Timezone, Currency, ISP, Organization, AS Number
    - Status flags (Mobile, Proxy, Hosting)
    - DNS resolver information

    Returns:
        dict: A dictionary containing the network details or an error.
    """
    if not os.path.exists(BINARY_PATH):
        return {
            "error": f"Binary not found at {BINARY_PATH}. Please compile the Go utility."
        }

    try:
        # Call the Go binary
        result = subprocess.run(
            [BINARY_PATH],
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
