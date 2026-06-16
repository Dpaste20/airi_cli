---
name: find-skills
description: Discover installed skills and install new ones from skills.sh. Use when the user asks 'find a skill for X', 'is there a skill for X', 'what skills do you have', 'install a skill for X', or expresses a need for a capability that might exist as an installable skill.
---

# Find Skills

This skill helps you discover what skills are currently installed in Airi and install new ones from the open skills.sh ecosystem.

## What Are Skills?

Skills are packages of domain expertise loaded from `./skills/`. Each skill is a folder containing a `SKILL.md` file with YAML frontmatter (`name`, `description`) and a body of procedural instructions. Agno's `Skills` system exposes them to you via three tools:

- `get_skill_instructions(skill_name)` — load a skill's full instruction body
- `get_skill_reference(skill_name, reference_path)` — read a reference document from a skill
- `get_skill_script(skill_name, script_path, execute=False)` — read or run a skill's script

You also have two Airi-native skill tools:

- `list_skills()` — list all currently installed skills with their names and descriptions
- `install_skill(skill_name, repo)` — install a skill from a GitHub repository into `./skills/`

## When to Use This Skill

- User asks "what skills do you have?" or "list your skills" → call `list_skills()`
- User asks "find a skill for X" or "is there a skill for X" → search skills.sh, then offer `install_skill()`
- User asks "install a skill for X" → find the best match on skills.sh and call `install_skill()`
- User asks "can you do X" where X is a specialised domain → check installed skills first, then skills.sh

## Workflow: Finding and Installing a New Skill

### Step 1 — Check installed skills first
Call `list_skills()`. If an installed skill already covers the need, call `get_skill_instructions(skill_name)` and proceed with it. Do not install a duplicate.

### Step 2 — Check skills.sh leaderboard
Browse https://skills.sh/ for well-known skills in the domain. Top sources to check:
- `vercel-labs/agent-skills` — React, Next.js, web design, browser automation
- `anthropics/skills` — frontend design, document generation (docx/pdf/pptx/xlsx)
- `obra/superpowers` — agent workflows, subagent delegation, debugging
- `microsoft/azure-skills` — Azure cloud services

### Step 3 — Verify quality before recommending
Do not recommend a skill based on a name alone. Always verify:
1. **Install count** — prefer 1K+ installs; be cautious below 100
2. **Source reputation** — `vercel-labs`, `anthropics`, `microsoft` are trusted; unknown authors require more scrutiny
3. **Match** — confirm the skill's description actually addresses the user's need

### Step 4 — Present to the user
When you find a relevant skill, tell the user:
- What the skill does
- Its install count and source repo
- Ask for confirmation before installing

Example:
```
I found a skill that might help. The "agent-browser" skill from vercel-labs/agent-browser
provides full browser automation — navigate, click, fill forms, extract data, screenshot.
(417K installs)

Shall I install it? I'll run:
install_skill("agent-browser", "vercel-labs/agent-browser")
```

### Step 5 — Install
If the user confirms, call:
```
install_skill(skill_name="agent-browser", repo="vercel-labs/agent-browser")
```
After a successful install, the skill is immediately available in `./skills/` and will be loaded on the next agent boot. Inform the user it will be active after Airi restarts.

## When No Skill Exists

If no relevant skill is found on skills.sh:
1. Say so clearly — do not hallucinate a skill name
2. Offer to help with the task directly using your native tools
3. Suggest the user could create their own skill: `npx skills init my-skill-name`

## Airi-Specific Notes

- Skills live in `./skills/<skill-name>/SKILL.md`
- The `Skills(loaders=[LocalSkills("./skills")])` call in `server.py` loads them at boot — a restart is required after `install_skill()`
- Skill names must be lowercase, alphanumeric + hyphens only, folder name must match the `name` field in frontmatter
- Do not load `get_skill_instructions` for every installed skill speculatively — only load the one relevant to the current task
