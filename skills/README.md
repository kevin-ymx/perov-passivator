# Agent Skills (perov-passivator)

Project-local [Cursor Agent Skills](https://cursor.com/docs/agent/skills) for this repository.
Each skill is a subdirectory with a required `SKILL.md` (instructions + YAML frontmatter) and
optional supporting files (`config_template.json`, `examples.md`, `scripts/`, etc.).

## Layout

```
skills/
├── README.md                 # this file
├── pubchem-mol-filter/       # RDKit filtering of PubChem CSV shards
├── ssl-neighbor-search/      # GIN-E SSL embedding nearest-neighbor search
└── mol-salt-vendor/          # LLM + web search: physical form & halide-salt vendors
```

Typical skill folder:

```
<skill-name>/
├── SKILL.md                  # required — agent instructions + frontmatter
├── examples.md               # optional — prompt → config examples
├── config_template.json      # optional — run config template
├── requirements.txt          # optional — dependencies
└── scripts/                  # optional — runnable helpers
```

## How agents use these skills

1. Read `SKILL.md` when the task matches the skill's `description`.
2. Build a **run config** from the user prompt + `config_template.json` (defaults for unspecified fields).
3. **Show the full config for approval** — all three skills gate on `"confirmed": true`.
4. Run the script under `scripts/` and report outputs.

Do **not** put custom skills in `~/.cursor/skills-cursor/` (Cursor built-ins only).

## Skills in this repo

| Skill | Purpose | Runs on |
|-------|---------|---------|
| [pubchem-mol-filter](pubchem-mol-filter/SKILL.md) | Filter PubChem CSV/shard data with configurable RDKit criteria | Local or HPC (Slurm) |
| [ssl-neighbor-search](ssl-neighbor-search/SKILL.md) | Nearest neighbors of user-given molecules (inline or CSV) in GIN-E SSL embedding space; emits a dedup `cid`/`smiles` table | Local or HPC node (direct `python` CLI; needs checkpoint + embeddings + PyTorch/RDKit) |
| [mol-salt-vendor](mol-salt-vendor/SKILL.md) | Per-molecule free-base physical form, vendors, and HCl/HBr/HI salt availability via OpenAI + web search | Local (OpenAI API) |

### Discovery funnel

The skills chain into a candidate-discovery pipeline:

```
pubchem-mol-filter → (ML ranking) → ssl-neighbor-search → mol-salt-vendor
   clean candidates                   analogs of actives     buyable forms + vendors
```

`ssl-neighbor-search` writes a deduplicated `cid`/`smiles` table that drops directly into
`mol-salt-vendor`'s default input columns.

## Add a new skill

1. Create `skills/<skill-name>/` (lowercase, hyphens only).
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

## Personal vs project skills

| Location | Scope |
|----------|--------|
| `skills/` (here) | This repo only — shared via git |
| `~/.cursor/skills/` | All your projects on this machine |

## Security

- Never commit API keys or secrets; use environment variables (e.g. `OPENAI_API_KEY`).
- Review generated configs before setting `"confirmed": true`.
