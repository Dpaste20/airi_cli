import json
import os
import re
import subprocess
from pathlib import Path

from agno.tools import tool

SKILLS_DIR = os.path.join(os.getcwd(), "skills")


def _parse_frontmatter(skill_md: Path) -> dict:
    """
    Extract name and description from a SKILL.md YAML frontmatter block.
    Returns {} if parsing fails.
    """
    try:
        content = skill_md.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return {}
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}

        result = {}
        for line in parts[1].strip().splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key in ("name", "description"):
                    result[key] = val
        return result
    except Exception:
        return {}


def _is_valid_skill_name(name: str) -> bool:
    """Validate against the Agent Skills spec naming rules."""
    return bool(re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$", name)) or bool(
        re.match(r"^[a-z0-9]$", name)
    )


def _resolve_repo_url(repo: str) -> str:
    """
    Accept either a bare 'owner/repo' slug or a full https:// URL.
    Returns a full GitHub HTTPS URL.
    """
    repo = repo.strip()
    if repo.startswith("https://"):
        return repo
    if re.match(r"^[\w\-]+/[\w\-]+$", repo):
        return f"https://github.com/{repo}"
    raise ValueError(f"Cannot resolve repo to a GitHub URL: {repo!r}")


@tool
def list_skills() -> str:
    """
    Lists all skills currently installed in Airi's ./skills/ directory.

    Returns each skill's name and description so you can decide whether
    an installed skill already covers a user's request before searching
    skills.sh for something new.

    Returns:
        A formatted JSON list of installed skills, or a message if none exist.
    """
    skills_path = Path(SKILLS_DIR)

    if not skills_path.exists():
        return json.dumps(
            {
                "installed_skills": [],
                "message": (
                    "No skills directory found at ./skills/. "
                    "Install your first skill with install_skill()."
                ),
            }
        )

    installed = []
    for item in sorted(skills_path.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        skill_md = item / "SKILL.md"
        if not skill_md.exists():
            continue
        fm = _parse_frontmatter(skill_md)
        installed.append(
            {
                "name": fm.get("name", item.name),
                "description": fm.get("description", "(no description)"),
                "path": str(item),
            }
        )

    if not installed:
        return json.dumps(
            {
                "installed_skills": [],
                "message": (
                    "The ./skills/ directory exists but contains no valid skills yet. "
                    "Use install_skill() to add one."
                ),
            }
        )

    return json.dumps(
        {
            "installed_skills": installed,
            "count": len(installed),
            "message": (
                f"{len(installed)} skill(s) installed. "
                "Use get_skill_instructions(name) to load a skill's full guidance."
            ),
        },
        indent=2,
    )


@tool
def install_skill(skill_name: str, repo: str) -> str:
    """
    Installs a skill from a GitHub repository into Airi's ./skills/ directory
    using the 'npx skills add' CLI from the skills.sh ecosystem.

    Always confirm with the user before calling this — installation fetches
    content from an external GitHub repository.

    The skill becomes available after Airi is restarted (the Skills loader
    runs at boot time via LocalSkills in server.py).

    Args:
        skill_name: The exact skill name as listed on skills.sh
                    (e.g. "agent-browser", "frontend-design").
                    Must be lowercase, alphanumeric and hyphens only.
        repo:       The GitHub repository slug or full URL containing the skill
                    (e.g. "vercel-labs/agent-browser" or
                    "https://github.com/anthropics/skills").

    Returns:
        A JSON object with status, message, and next steps.

    Example:
        install_skill("agent-browser", "vercel-labs/agent-browser")
        install_skill("frontend-design", "anthropics/skills")
    """
    skill_name = skill_name.strip().lower()

    if not _is_valid_skill_name(skill_name):
        return json.dumps(
            {
                "status": "error",
                "message": (
                    f"Invalid skill name '{skill_name}'. "
                    "Names must be lowercase, alphanumeric + hyphens, "
                    "cannot start or end with a hyphen."
                ),
            }
        )

    # Resolve repo to a full URL
    try:
        repo_url = _resolve_repo_url(repo)
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})

    # Guard: don't reinstall if already present
    target_dir = Path(SKILLS_DIR) / skill_name
    if target_dir.exists() and (target_dir / "SKILL.md").exists():
        return json.dumps(
            {
                "status": "already_installed",
                "skill_name": skill_name,
                "path": str(target_dir),
                "message": (
                    f"Skill '{skill_name}' is already installed at {target_dir}. "
                    "Use get_skill_instructions(skill_name) to load it."
                ),
            }
        )

    os.makedirs(SKILLS_DIR, exist_ok=True)

    cmd = [
        "npx",
        "--yes",
        "skills",
        "add",
        repo_url,
        "--skill",
        skill_name,
        "--copy",
        "--dir",
        SKILLS_DIR,
        "-y",
    ]

    print(f"[SkillsTools] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd(),
        )
    except FileNotFoundError:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    "npx not found. Node.js must be installed to use install_skill(). "
                    "Alternatively, manually place a SKILL.md in ./skills/<skill-name>/."
                ),
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "status": "error",
                "message": "install_skill timed out after 120s. Check your network connection.",
            }
        )

    if result.returncode != 0:
        return json.dumps(
            {
                "status": "error",
                "skill_name": skill_name,
                "repo": repo,
                "message": "npx skills add failed.",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )

    if not (target_dir / "SKILL.md").exists():
        return json.dumps(
            {
                "status": "error",
                "skill_name": skill_name,
                "message": (
                    f"npx skills add exited 0 but SKILL.md not found at {target_dir}. "
                    "The skill name may not exist in that repository. "
                    "Check https://skills.sh for the exact name and repo."
                ),
                "stdout": result.stdout.strip(),
            }
        )

    return json.dumps(
        {
            "status": "success",
            "skill_name": skill_name,
            "path": str(target_dir),
            "message": (
                f"Skill '{skill_name}' installed at {target_dir}. "
                "It will be active after Airi restarts. "
                "Restart the server to load it, then call "
                f"get_skill_instructions('{skill_name}') to use it."
            ),
        }
    )
