import asyncio
import os

from agno.tools import tool


async def _run_single_notion(args: str, env: dict) -> tuple[bool, str]:
    """
    Runs a single ntn (Notion CLI) command.
    Returns (success, output_string).
    """
    # Ensure the command starts with ntn if the agent didn't include it
    command_str = args.strip()
    if command_str.startswith("ntn "):
        full_command = command_str
    else:
        full_command = f"ntn {command_str}"

    print(f"Executing Notion Command: {full_command}")

    # Standard timeout for API requests
    timeout = 30

    try:
        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return False, f"Error: '{full_command}' timed out (limit: {timeout}s)."

        output = stdout.decode().strip()
        error_msg = stderr.decode().strip()

        if process.returncode == 0:
            return (
                True,
                output
                if output
                else f"✓ Command executed successfully: {full_command}",
            )
        else:
            return False, (
                f"Error (Exit Code {process.returncode}):\n"
                f"Stdout: {output}\n"
                f"Stderr: {error_msg}"
            )

    except Exception as e:
        return False, f"Notion CLI Execution Exception: {str(e)}"


@tool
async def notion(args: str) -> str:
    """
    Controls a Notion workspace by executing commands via the 'ntn' (Notion CLI) tool.

    WORKFLOW
    --------
    Use this tool to interact directly with the Notion API, read/write Markdown,
    upload files, and manage Notion Workers.

    PAGES (Markdown)
    ----------------
    - "pages get <page-id>"             Retrieves a Notion page and converts it to clean Markdown.
    - "pages create --parent page:<id> --content '## Hello'"  Creates a page with Markdown content.

    API REQUESTS (Direct JSON interaction)
    --------------------------------------
    - "api ls"                          List all available API endpoints.
    - "api <endpoint> --help"           Get help and property syntax for a specific endpoint.
    - "api v1/users"                    GET request to list users.
    - "api v1/pages parent[page_id]=abc properties[title][title][0][text][content]='My Page'"
                                        POST request to create a page using inline JSON construction.
    - "api v1/pages/abc -X PATCH archived:=true"
                                        PATCH request to update/archive a page.
    - "api v1/databases/abc/query filter[property]='Status' filter[select][equals]='Done'"
                                        POST request to query a database.

    FILES & UPLOADS
    ---------------
    - "files create < photo.png"        Upload a local file to Notion hosting.
    - "files create --external-url <url>" Create a file reference from an external URL.
    - "files list"                      List recently uploaded files.

    NOTION WORKERS
    --------------
    - "workers new <name>"              Scaffold a new Notion Worker project.
    - "workers deploy"                  Build and upload the Worker from the current directory.
    - "workers list"                    List deployed workers.
    - "workers logs <worker-id>"        Tail logs for a deployed worker.

    BATCH EXECUTION EXAMPLE
    -----------------------
    "pages get abc12345"
    "api v1/pages/abc12345 -X PATCH archived:=true"

    Args:
        args (str): Arguments to pass to the ntn CLI (e.g., "pages get abc123", "api v1/users").
                    You can omit the 'ntn' prefix; it will be added automatically.
    """
    env = os.environ.copy()

    commands = [
        line.strip()
        for line in args.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not commands:
        return "Error: No commands provided."

    if len(commands) == 1:
        _, result = await _run_single_notion(commands[0], env)
        return result

    results: list[str] = []
    for i, cmd in enumerate(commands, 1):
        success, output = await _run_single_notion(cmd, env)
        results.append(f"[{i}] {cmd}\n{output}")
        if not success:
            remaining = len(commands) - i
            if remaining:
                results.append(
                    f"⚠ Aborted: {remaining} command(s) not executed due to previous failure."
                )
            break

    return "\n\n".join(results)
