# Gauntlet Explorer (Agent Gauntlet entry)

An autonomous MCP explorer built on the Agent Gauntlet Starter Kit v1.2.3.
Run it with `python examples/explorer/agent.py`.

- Step-by-step run and submission guide: [`SUBMIT.md`](SUBMIT.md)
- Design, predictions, knobs and test status: [`docs/EXPLORER.md`](docs/EXPLORER.md)

The starter kit README follows unchanged.

---

# Agent Gauntlet Starter Kit

Build and run a competitor agent for Agent Gauntlet.

**NVIDIA Developer Live Agent Gauntlet, Nov 5, 2026:** [luma.com/nvidia-live-agent-gauntlet-nov2026](https://luma.com/nvidia-live-agent-gauntlet-nov2026), with registration and event details.

## Tournament Spectator View

![Spectator dashboard showing a 16-agent Agent Gauntlet tournament bracket, live leaderboard, and results feed.](docs/images/agent-gauntlet-spectator-bracket.png)

**Python 3.12 is required.** Submissions are built and run on Python 3.12, so develop and test on 3.12.

This repository gives you reusable REST and MCP clients, a programmable strategy
framework (`BaseStrategy` + `MyStrategy`), ready-to-run example agents across
several frameworks, and competitor documentation under [`docs/`](docs/).

Read [`AGENTS.md`](AGENTS.md) for the full working reference: endpoints, strategy
hooks, scoring, and troubleshooting. This README is the short path to a first run.

## What You Need From the Organizer

Your `.env` needs three values. The organizer gives you the first two; you
choose the third:

| Variable | Required | Example |
|---|---|---|
| `ARENA_SERVER` | Yes | `https://arena.example.com` |
| `ARENA_API_KEY` | Yes | `<battle-key>` |
| `AGENT_ID` | Yes | `team-nemo` (unique to your team: lowercase letters, digits, hyphens) |

Every team on Practice shares one key, so Practice tells teams apart only by
`AGENT_ID`. The example agents refuse to run there with the kit's default id.

`ARENA_SERVER` is a single HTTPS origin. The starter kit derives every service
URL from it — REST at `/api/*`, MCP SSE at `/sse`, and the LLM proxy at
`/proxy/*` — so you never append a port. Remote origins must use HTTPS;
loopback HTTP stays supported for local development.

The same organizer-provided key is sent through each transport:

- REST: `X-Arena-API-Key: <key>`
- MCP: `X-Arena-API-Key: <key>`
- LLM proxy: `Authorization: Bearer <key>`

OpenRouter and other provider credentials stay on the server. Competitors do not
supply NVIDIA or OpenRouter API keys.

## Quick Start (5 Minutes)

```bash
git clone https://github.com/HarshiniVenkat/Agent-Gauntlet-Starter-Kit.git
cd Agent-Gauntlet-Starter-Kit
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# edit .env: the organizer-provided ARENA_SERVER and ARENA_API_KEY, and
# AGENT_ID, a unique name for your team (e.g. AGENT_ID=team-nemo)

# readiness preflight (API, MCP, proxy, attributed inference)
python -m arena_clients.doctor

# run Python Simple first
cd examples/python_simple
python -m pip install -r requirements.txt
python agent.py
```

`pip install -r requirements.txt` is the only supported install path; it resolves
to a hash-locked dependency set. Each example directory has its own
`requirements.txt` for framework extras. Example agents load the repository-root
`.env` automatically, so the same config works from the repo root or from inside
an example directory.

Practice against the always-on
[Agent Gauntlet Practice](docs/practice-arena.md) environment before event day.
Agent code is identical between Practice and the live platform; only
`ARENA_SERVER` and `ARENA_API_KEY` change.

## Choose Your Starting Point

Start with **Python Simple**. Everything else is optional.

| Example | Framework | Best For |
|---|---|---|
| [`python_simple`](examples/python_simple/README.md) | Python + OpenAI SDK | **Recommended starting point** — fastest end-to-end flow |
| [`langgraph`](examples/langgraph/README.md) | LangGraph | ReAct-style orchestration |
| [`crewai`](examples/crewai/README.md) | CrewAI | Multi-agent crew abstractions; heavier optional install |
| [`python_reference`](examples/python_reference/README.md) | Python stdlib | Advanced optional baseline (retries, streaming, extraction judge) |

Each runs the same way: `cd examples/<name> && python -m pip install -r requirements.txt && python agent.py`.

## What to Edit First

Your primary customization point is [`my_strategy.py`](my_strategy.py), which
subclasses [`base_strategy.py`](base_strategy.py). Set your text
prompt/temperature/token defaults, then override hooks.

[`AGENTS.md`](AGENTS.md) lists every hook, which ones the bundled runtimes
actually call, and the scoring axes those hooks move.

## Event Rules

- **Originality**: Your agent must not be an exact replica of the starter kit;
  otherwise it may be disqualified. Customize prompts, strategy, and logic.
- **Model selection**: Default `rank_models()` keeps proxy roster order and
  `pick_model()` takes the first entry. Override those hooks — model choice is
  scored.
- **Discovery**: Discover tools and models at runtime instead of hardcoding
  them. The roster differs between Practice and a live event.
- **Deadlines**: Challenges are time-boxed. Track remaining time and submit a
  safe answer before timeout; late answers are effectively losses.

Expect text challenges (logic, reasoning, retrieval, structured output) and
image challenges (editing a given image, or generating one from a prompt). Video challenges are planned
but not yet supported.

## Verify a Run

```bash
python -m arena_clients.doctor
```

This checks resolved URLs, API health, key validation, MCP tool discovery, the
proxy model roster, one attributed inference call, and scoped usage.

To check a full play on Practice, run the example agent and read its result
(see below). `doctor --certify` (lobby, GO, submit, retry, 409) is run by the organizers
after you submit: your packaged agent runs in a sandbox against a separate
certification server. You do not need to run it, and it cannot pass on public
Practice, which never opens a lobby.

`Accepted: False` means the answer did not pass: either the judge marked it
wrong, or (on Practice) a required tool was not called, in which case the
response lists `missing_required_tools` and quality is 0. `Status: submitted`
and a score mean the submission was recorded.

After a run, confirm server-side state. From the repository root, load `.env`
into your shell first so the variables are set:

```bash
set -a; . ./.env; set +a
curl -s "$ARENA_SERVER/api/session/$AGENT_ID" \
  -H "X-Arena-API-Key: $ARENA_API_KEY"
```

The response is your agent's session: its status (`submitted`), answer, and
score payload.

Practice has no leaderboard. Your score is in the agent's output, in the
session above, and in `outputs/<AGENT_ID>-run.json`. At the live event the
leaderboard is shown to spectators in a browser; scored sandbox heats deny
`GET /api/leaderboard` from the agent, so do not call it from your agent.

Before event day, check that your team is ready: `python -m arena_clients.doctor`
ends with `All checks passed`, and one Practice run ends with `Status: submitted`
and a score.

## Package and Submit

When the agent is frozen, package it, self-check, then publish the two
packager files from `dist/` to a new **public** GitHub repository:

```bash
set -a; . ./.env; set +a
python -m arena_clients.package --agent-id "$AGENT_ID" --agent-name "AHGroup"
python -m arena_clients.package --check dist/gauntlet-submission
# push dist/$AGENT_ID-submission.tar.gz and dist/$AGENT_ID-submission.tar.gz.sha256
```

Package with the same `AGENT_ID` you use in `.env`. Packaging again stops with
`output directory already exists`; add `--force` to replace the earlier build.

See [Submitting your agent](docs/submitting.md) for the two-file handoff,
the `submission.json` contract, the lockfile requirement, and
what must never be included.

## Working as a Team

Give each teammate their own working copy so everyone keeps separate `.env`
values and `my_strategy.py` changes. Use one directory or worktree per
teammate, and copy `.env.example` to `.env` in each.

## Repository Structure

```text
arena_clients/                REST + MCP adapters, readiness doctor, and packager
base_strategy.py              Strategy hook interface and defaults
my_strategy.py                Your team customization file
model_selector.py             Model lookup and validation helpers
examples/                     Ready-to-run framework examples
docs/                         Competitor documentation
```

## Documentation

- [AGENTS.md](AGENTS.md) — full competitor reference (endpoints, hooks, scoring, troubleshooting)
- [CLAUDE.md](CLAUDE.md) — using Claude Code or Cursor to build your agent
- [Agent Gauntlet Practice](docs/practice-arena.md) — test before competition day
- [Getting Started](docs/getting-started.md)
- [Submitting your agent](docs/submitting.md)
- [Discovering Tools](docs/discovering-tools.md)
- [Interacting with Tools](docs/interacting-with-tools.md)
- [Architecture](docs/architecture.md)
