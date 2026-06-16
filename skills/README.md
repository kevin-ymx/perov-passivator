# LBPP Agent Skills

Project-local [Cursor Agent Skills](https://cursor.com/docs/agent/skills) for this repository.
Each skill is a subdirectory with a required `SKILL.md` file.

## Layout

```
.cursor/skills/
├── README.md                 # this file
└── <skill-name>/
    ├── SKILL.md              # required — instructions + YAML frontmatter
    ├── reference.md          # optional — detailed docs
    ├── examples.md           # optional — usage examples
    └── scripts/              # optional — helper scripts
```

## Add a new skill

1. Create a folder: `.cursor/skills/<skill-name>/` (lowercase, hyphens only).
2. Add `SKILL.md` with frontmatter:

```markdown
---
name: skill-name
description: What it does and when the agent should use it (third person, trigger terms).
---

# Skill title

## Instructions
...
```

3. Commit the folder so collaborators get the same agent behavior.

Do **not** put custom skills in `~/.cursor/skills-cursor/` (Cursor built-ins only).

## Personal vs project skills

| Location | Scope |
|----------|--------|
| `.cursor/skills/` (here) | This repo only — shared via git |
| `~/.cursor/skills/` | All your projects on this machine |

## Skills in this repo

| Skill | Purpose |
|-------|---------|
| [lbpp-pubchem-molecule-filter](lbpp-pubchem-molecule-filter/SKILL.md) | Configure and run RDKit filters on PubChem CSV / shard data from user-defined criteria |
