import asyncio
import os

from agno.tools import tool

_session_started: bool = False


OBSERVATION_COMMANDS = (
    # Page inspection
    "snapshot",
    "screenshot",
    "pdf ",
    "eval ",
    # Element queries
    "get ",
    "is ",
    "find ",
    # Network / storage
    "cookies",
    "storage",
    "network requests",
    "network request ",
    "network har",
    # State / session
    "state list",
    "state show",
    "session",
    # Tabs
    "tab",
    # Auth vault
    "auth list",
    "auth show",
    # Debug / diagnostics
    "console",
    "errors",
    "stream status",
    "profiles",
    "doctor",
    # Blocking waits (return on condition met)
    "wait ",
    # Batch (may contain any mix of observation/action commands)
    "batch",
    # Clipboard read
    "clipboard read",
    # React / vitals inspection
    "react ",
    "vitals",
)


def _needs_output(args: str) -> bool:
    """Returns True if the command produces output that must be read."""
    stripped = args.strip().lower()
    return any(stripped.startswith(cmd) for cmd in OBSERVATION_COMMANDS)


def _resolve_timeout(args: str, wait_for_output: bool) -> int:
    """
    Returns the appropriate timeout in seconds.

    - wait/batch/doctor: 180s (can be long-running)
    - Other observation commands: 120s
    - Fire-and-forget actions: 15s
    """
    stripped = args.strip().lower()
    if any(stripped.startswith(cmd) for cmd in ("wait ", "batch", "doctor")):
        return 180
    return 120 if wait_for_output else 15


async def _close_stale_session(env: dict) -> None:
    """Closes any stale agent-browser daemon/session from previous runs."""
    print("Closing any stale agent-browser session...")
    try:
        process = await asyncio.create_subprocess_shell(
            "agent-browser close",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        await asyncio.wait_for(process.communicate(), timeout=10)
    except Exception:
        pass


async def _run_single(args: str, env: dict) -> tuple[bool, str]:
    """
    Runs a single agent-browser command.
    Returns (success, output_string).
    """
    full_command = f"agent-browser --headed {args}"
    print(f"Executing Browser Command: {full_command}")

    wait_for_output = _needs_output(args)
    timeout = _resolve_timeout(args, wait_for_output)

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
            if not wait_for_output:
                return True, f"✓ Done: {args}"
            return False, f"Error: '{args}' timed out (limit: {timeout}s)."

        output = stdout.decode().strip()
        error_msg = stderr.decode().strip()

        if process.returncode == 0:
            if not wait_for_output:
                return True, f"✓ Done: {args}"
            return (
                True,
                output if output else "Command executed successfully (no output).",
            )
        else:
            return False, (
                f"Error (Exit Code {process.returncode}):\n"
                f"Stdout: {output}\n"
                f"Stderr: {error_msg}"
            )

    except Exception as e:
        return False, f"Browser Execution Exception: {str(e)}"


@tool
async def agent_browser(args: str) -> str:
    """
    Controls a web browser to navigate, interact, and extract data using the 'agent-browser' CLI.

    Runs in native mode (direct CDP via Rust binary) with a persistent profile and headed
    (visible) browser window. Configured via environment variables injected per command:
      AGENT_BROWSER_NATIVE=1
      AGENT_BROWSER_PROFILE=./airi_browse_dir

    CRITICAL — BATCHING RULE
    ------------------------
    NEVER call this tool once per command. ALWAYS pass all sequential commands
    together in a single call, separated by newlines. The tool executes them in
    order and returns combined output.


    The ONLY reason to make a second call is when you need to read snapshot
    output (e.g. element refs) before deciding what to interact with next.

    WORKFLOW
    --------
    1. Open a URL:        "open <url>"
    2. Snapshot elements: "snapshot -i"
    3. Interact via refs: "click @e1", "fill @e2 'text'"
    4. Re-snapshot after page changes.

    NAVIGATION
    ----------
    - "open <url>"               Navigate to URL (aliases: goto, navigate)
    - "open"                     Launch browser on about:blank (for pre-nav setup)
    - "back" / "forward"         Browser history
    - "reload"                   Refresh current page
    - "pushstate <url>"          SPA client-side nav (auto-detects Next.js router)

    SNAPSHOTS & SCREENSHOTS
    -----------------------
    - "snapshot -i"              Accessibility tree with element refs (@e1, @e2...)
    - "screenshot [path]"        Screenshot (--full for full page)
    - "screenshot --annotate"    Annotated screenshot with numbered element labels
    - "pdf <path>"               Save page as PDF

    INTERACTION (use @eN refs from snapshot)
    -----------------------------------------
    - "click @e1"                Click element (--new-tab to open in new tab)
    - "dblclick @e1"             Double-click
    - "fill @e1 'text'"          Clear and fill input
    - "type @e1 'text'"          Type into element
    - "press Enter"              Press key (Enter, Tab, Control+a, etc.)
    - "keyboard type 'text'"     Type at current focus (no selector)
    - "keyboard inserttext 'x'"  Insert text without triggering key events
    - "hover @e1"                Hover element
    - "focus @e1"                Focus element
    - "select @e1 'option'"      Select dropdown option
    - "check @e1"                Check checkbox
    - "uncheck @e1"              Uncheck checkbox
    - "scroll down 500"          Scroll (up/down/left/right, --selector @e1)
    - "scrollintoview @e1"       Scroll element into view
    - "drag @e1 @e2"             Drag and drop
    - "upload @e1 /path/to/file" Upload file(s)

    SEMANTIC FINDERS (when you don't have a ref)
    ---------------------------------------------
    - "find role button click --name 'Submit'"
    - "find label 'Email' fill 'test@test.com'"
    - "find text 'Login' click"
    - "find placeholder 'Search' fill 'query'"
    - "find alt 'Logo' click"
    - "find testid 'submit-btn' click"
    - "find first '.item' click"
    - "find last '.item' text"
    - "find nth 2 '.card' hover"
    Options: --name <name>, --exact

    GET INFO
    --------
    - "get text @e1"             Text content
    - "get html @e1"             innerHTML
    - "get value @e1"            Input value
    - "get attr @e1 href"        Attribute
    - "get title"                Page title
    - "get url"                  Current URL
    - "get count .selector"      Count matching elements
    - "get box @e1"              Bounding box
    - "get styles @e1"           Computed styles

    CHECK STATE
    -----------
    - "is visible @e1"
    - "is enabled @e1"
    - "is checked @e1"

    WAIT
    ----
    - "wait 3000"                Wait ms
    - "wait @e1"                 Wait for element
    - "wait --text 'Welcome'"    Wait for text
    - "wait --url '**/dash'"     Wait for URL pattern
    - "wait --load networkidle"  Wait for network idle
    - "wait --fn 'condition'"    Wait for JS condition
    - "wait --download [path]"   Wait for download
    - "wait '#spinner' --state hidden"  Wait for element to disappear

    TABS & FRAMES
    -------------
    - "tab"                      List tabs (tabId + label)
    - "tab new [url]"            New tab
    - "tab new --label docs <url>"  Named tab
    - "tab docs"                 Switch to tab by label or id (t1, t2...)
    - "tab close [t1|label]"     Close tab
    - "frame @e3"                Switch into iframe
    - "frame main"               Return to main frame

    NETWORK
    -------
    - "network route <url> --abort"           Block requests
    - "network route <url> --body <json>"     Mock response
    - "network route '*' --abort --resource-type script"
    - "network unroute [url]"                 Remove route
    - "network requests"                      View tracked requests
    - "network requests --filter <pattern>"   Filter by URL
    - "network requests --type xhr,fetch"     Filter by type
    - "network requests --method POST"        Filter by method
    - "network requests --status 2xx"         Filter by status
    - "network har start"                     Start HAR recording
    - "network har stop [output.har]"         Stop and save HAR

    COOKIES & STORAGE
    -----------------
    - "cookies"                  Get all cookies
    - "cookies set <name> <val>" Set cookie
    - "cookies clear"            Clear cookies
    - "storage local"            Get all localStorage
    - "storage local <key>"      Get specific key
    - "storage local set k v"    Set value
    - "storage local clear"      Clear all
    - "storage session ..."      Same for sessionStorage

    CLIPBOARD
    ---------
    - "clipboard read"           Read clipboard text
    - "clipboard write 'text'"   Write to clipboard
    - "clipboard copy"           Copy selection (Ctrl+C)
    - "clipboard paste"          Paste (Ctrl+V)

    SETTINGS
    --------
    - "set viewport 1280 800"    Set viewport
    - "set device 'iPhone 14'"   Emulate device
    - "set geo <lat> <lng>"      Set geolocation
    - "set offline on"           Toggle offline mode
    - "set headers <json>"       Extra HTTP headers
    - "set media dark"           Emulate color scheme

    MOUSE
    -----
    - "mouse move <x> <y>"
    - "mouse down [button]"
    - "mouse up [button]"
    - "mouse wheel <dy> [dx]"

    DIALOGS
    -------
    - "dialog accept [text]"     Accept dialog
    - "dialog dismiss"           Dismiss dialog
    - "dialog status"            Check if dialog is open

    AUTH VAULT
    ----------
    - "auth save <name> --url <url> --username <u> --password <p>"
    - "auth login <name>"        Login using saved credentials
    - "auth list"                List saved profiles
    - "auth show <name>"         Show profile metadata

    STATE MANAGEMENT
    ----------------
    - "state save <path>"        Save auth state
    - "state load <path>"        Load auth state
    - "state list"               List saved state files
    - "state show <file>"        Show state summary

    DEBUG
    -----
    - "console"                  View console messages
    - "console --clear"          Clear console log
    - "errors"                   View page errors
    - "errors --clear"           Clear error log
    - "highlight @e1"            Highlight element
    - "eval '<js>'"              Run JavaScript
    - "doctor"                   Diagnose install

    REACT / WEB VITALS
    ------------------
    - "open --enable react-devtools <url>"
    - "react tree"               Component tree
    - "react inspect <fiberId>"  Inspect component
    - "vitals [url]"             LCP/CLS/TTFB/FCP/INP

    BATCH EXECUTION
    ---------------
    - "batch 'open https://example.com' 'snapshot -i' 'screenshot'"
    - "batch --bail 'open <url>' 'click @e1'"
    Pipe JSON via stdin for complex batches.

    STREAMING
    ---------
    - "stream enable [--port 9223]"  Start WebSocket streaming
    - "stream status"                Check streaming state
    - "stream disable"               Stop streaming

    PROFILES & SESSIONS
    -------------------
    - "profiles"                 List Chrome profiles
    - "session"                  Show current session
    - "session list"             List active sessions

    Args:
        args (str): Arguments to pass to agent-browser CLI (e.g., "open google.com", "click @e1").
    """
    global _session_started

    env = os.environ.copy()
    env["AGENT_BROWSER_NATIVE"] = "1"
    env["AGENT_BROWSER_PROFILE"] = os.path.abspath("airi_browse_dir")

    if not _session_started:
        await _close_stale_session(env)
        _session_started = True

    commands = [
        line.strip()
        for line in args.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not commands:
        return "Error: No commands provided."

    if len(commands) == 1:
        _, result = await _run_single(commands[0], env)
        return result

    results: list[str] = []
    for i, cmd in enumerate(commands, 1):
        success, output = await _run_single(cmd, env)
        results.append(f"[{i}] {cmd}\n{output}")
        if not success:
            remaining = len(commands) - i
            if remaining:
                results.append(f"⚠ Aborted: {remaining} command(s) not executed.")
            break

    return "\n\n".join(results)
