import asyncio
import re
import shlex

from agno.tools import tool

# ── Dangerous command patterns ─────────────────────────────────────────────
# Each entry is a regex tested against the *normalised* command string.
# Keep patterns specific enough to avoid false positives.

_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    # Disk / filesystem destruction
    (
        r"\brm\s+.*-[a-z]*r[a-z]*f|rm\s+.*-[a-z]*f[a-z]*r",
        "recursive force-delete (rm -rf)",
    ),
    (r"\bdd\b.*of=/dev/", "raw disk write via dd"),
    (r"\bmkfs\b", "filesystem format (mkfs)"),
    (r"\bshred\b", "secure file wipe (shred)"),
    (r">\s*/dev/s[a-z]+", "direct block device overwrite"),
    (r">\s*/dev/nvme", "direct NVMe device overwrite"),
    # System state
    (r"\b(shutdown|poweroff|halt|reboot|init\s+[06])\b", "system shutdown/reboot"),
    (
        r"\bsystemctl\s+(stop|disable|mask)\s+(ssh|network|firewall|ufw|iptables)",
        "disabling critical services",
    ),
    (r"\bkillall\b|\bpkill\s+-9\s+-u\b", "mass process kill"),
    # Privilege escalation
    (r"\bchmod\s+[0-7]*777\b|\bchmod\s+-R\s+777\b", "world-writable chmod 777"),
    (r"\bchown\s+-R\b.*:/", "recursive ownership change on root paths"),
    (
        r"\bsudo\s+su\b|\bsudo\s+-i\b|\bsudo\s+bash\b|\bsudo\s+sh\b",
        "sudo shell escalation",
    ),
    # Fork bombs and resource exhaustion
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),
    (r"\byes\b\s*\|", "yes-pipe resource abuse"),
    (r"/dev/zero|/dev/urandom.*>.*\bdd\b", "zero/random fill attack"),
    # Exfiltration / reverse shells
    (r"\b(nc|ncat|netcat)\b.*(-e|-c)\b", "netcat reverse shell"),
    (r"\bbash\s+-i\b.*>&\s*/dev/tcp/", "bash TCP reverse shell"),
    (r"\bcurl\b.*\|\s*(bash|sh|python|perl|ruby)\b", "curl-pipe-execute"),
    (r"\bwget\b.*-O\s*-.*\|\s*(bash|sh|python|perl|ruby)\b", "wget-pipe-execute"),
    # Crontab / persistence tampering
    (r"\bcrontab\s+-r\b", "delete all cron jobs"),
    (r"\becho\b.*>>\s*/etc/cron", "crontab injection"),
    # /etc and sensitive config tampering
    (
        r">\s*/etc/passwd|>\s*/etc/shadow|>\s*/etc/sudoers",
        "overwrite critical system files",
    ),
    (r"\bvisudo\b", "sudoers edit"),
    # History wiping (anti-forensics)
    (r">\s*~/\.bash_history|>\s*~/\.zsh_history", "shell history wipe"),
    (r"\bunset\s+HISTFILE\b", "disable shell history"),
]

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), label)
    for pattern, label in _BLOCKED_PATTERNS
]

# ── Shell injection characters that shouldn't appear in safe commands ───────
# Allow pipes (|) and redirects (>) since many legitimate commands use them,
# but block the most dangerous chaining forms.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (
        r";\s*(rm|dd|mkfs|shred|shutdown|reboot|curl\s.*\|\s*(bash|sh)|wget.*\|\s*(bash|sh))",
        "command chaining into destructive op",
    ),
    (
        r"`[^`]*(rm|dd|curl.*\||wget.*\|)[^`]*`",
        "backtick injection with dangerous command",
    ),
    (
        r"\$\([^)]*?(rm|dd|curl.*\||wget.*\|)[^)]*\)",
        "subshell injection with dangerous command",
    ),
]

_INJECTION_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), label)
    for p, label in _INJECTION_PATTERNS
]


def _check_command(command: str) -> str | None:
    """
    Returns a human-readable reason if the command should be blocked,
    or None if it is considered safe to run.
    """
    for pattern, label in _COMPILED:
        if pattern.search(command):
            return label

    for pattern, label in _INJECTION_COMPILED:
        if pattern.search(command):
            return label

    return None


@tool
async def bash(command: str) -> str:
    """
    Executes a shell command asynchronously.

    Certain destructive or dangerous commands are blocked and will return
    an error rather than execute.
    """
    print(f"Executing Shell Command: {command}")

    block_reason = _check_command(command)
    if block_reason:
        return (
            f"Blocked: This command was not executed.\n"
            f"Reason: Matched dangerous pattern — {block_reason}.\n"
        )

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return "Error: Command execution timed out (limit: 120s)."

        output = stdout.decode().strip()
        error_msg = stderr.decode().strip()

        if process.returncode == 0:
            return output if output else "Command executed successfully (no output)."
        else:
            return f"Error (Exit Code {process.returncode}): {error_msg}"

    except Exception as e:
        return f"Execution Exception: {str(e)}"
