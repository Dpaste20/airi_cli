import json

from agno.tools import tool

from sub_agents.manager import SubAgentManager, dumps_result
from utils.FetchUrls import fetch_urls
from utils.FileSearch import file_search
from utils.GetActiveConnections import get_active_connections
from utils.GetBatteryStatus import get_battery_status
from utils.GetDateTime import get_current_datetime
from utils.GetDiskSpace import get_disk_space
from utils.GetIPInfo import get_ip_info
from utils.GetRunningProcesses import get_running_processes
from utils.GetSystemLogs import get_system_logs
from utils.GetUptime import get_uptime
from utils.RagSearch import rag_search_tool
from utils.RunDiagnosticTool import run_system_diagnostic
from utils.SystemInfo import get_system_info


SUBAGENT_TOOL_REGISTRY = {
    "fetch_urls": fetch_urls,
    "file_search": file_search,
    "get_active_connections": get_active_connections,
    "get_battery_status": get_battery_status,
    "get_current_datetime": get_current_datetime,
    "get_disk_space": get_disk_space,
    "get_ip_info": get_ip_info,
    "get_running_processes": get_running_processes,
    "get_system_logs": get_system_logs,
    "get_uptime": get_uptime,
    "rag_search_tool": rag_search_tool,
    "run_system_diagnostic": run_system_diagnostic,
    "get_system_info": get_system_info,
}


def get_subagent_manager() -> SubAgentManager:
    return SubAgentManager(tool_registry=SUBAGENT_TOOL_REGISTRY)


@tool
def list_subagents() -> str:
    """
    List the configured local Airi sub-agents available for delegation.

    Use this before delegating if you are unsure which specialist should handle
    a task. The result includes each agent's description, model, and tool
    allowlist.
    """
    manager = get_subagent_manager()
    agents = [agent.as_dict() for agent in manager.list_agents()]

    return json.dumps(
        {
            "subagents": agents,
            "count": len(agents),
            "message": (
                "Use delegate_to_subagent(agent_name, task, context, session_id) "
                "to route a bounded task to one specialist."
            ),
        },
        indent=2,
    )


@tool
async def delegate_to_subagent(
    agent_name: str,
    task: str,
    context: str = "",
    session_id: str = "default_session",
) -> str:
    """
    Delegate a bounded task to a configured local Airi sub-agent.

    Args:
        agent_name: Exact sub-agent name, such as "alpha" or "beta".
        task: The specific task the sub-agent should complete.
        context: Optional relevant context from the current conversation.
        session_id: Optional stable session id for the delegated conversation.

    Returns:
        JSON with status, summary, full result, and error fields.
    """
    manager = get_subagent_manager()
    result = await manager.delegate(
        agent_name=agent_name.strip(),
        task=task,
        context=context,
        session_id=session_id,
    )
    return dumps_result(result)
