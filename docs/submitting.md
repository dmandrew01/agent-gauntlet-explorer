# Submitting your agent

The organizer admits a repository that holds the packager tarball and its
checksum. This kit assembles the submission tree, writes those two files,
and runs the same local checks the organizer runs before it builds an image.

Develop against Practice first. When the agent is frozen, run the
connectivity preflight and one full example run on Practice, package,
self-check, then publish a new repository with the two packager files. Send
the organizer that repository URL.

```text
develop  →  python -m arena_clients.doctor
         →  python examples/<your-example>/agent.py          # one full run on Practice
         →  python -m arena_clients.package --agent-id "$AGENT_ID" --agent-name "Your Team Name"
         →  python -m arena_clients.package --check dist/gauntlet-submission
         →  publish dist/<agent-id>-submission.tar.gz and dist/<agent-id>-submission.tar.gz.sha256
```

## The repository you owe

A **new public GitHub repository** whose default branch has these two files
at its root, named exactly as the packager wrote them:

```text
<agent-id>-submission.tar.gz
<agent-id>-submission.tar.gz.sha256
```

Use exactly one tarball and its checksum. Other files in the repository, such
as a README GitHub adds, are ignored, but a second tarball makes the
submission ambiguous and it is rejected.

Example for agent id `team-nemo`:

```text
team-nemo-submission.tar.gz
team-nemo-submission.tar.gz.sha256
```

This is not a fork of the starter kit. Do not push the unpacked
`dist/gauntlet-submission/` tree as the repository root. Do not send a lone
`my_strategy.py`. Do not use the host's auto-generated source zip (GitHub
"Download ZIP"). Organizers download only the
named tarball and checksum pair.

Python Simple, Python Reference, LangGraph, and CrewAI all use the same
two-file repository root. `--entrypoint` (and `--include` / `--lockfile` when
needed) only changes what is **inside** the tarball.

LangGraph example:

```bash
python -m arena_clients.package --agent-id team-nemo --agent-name "Team Nemo" \
  --entrypoint examples/langgraph/agent.py
```

CrewAI imports a helper module and needs the full dependency lock, so pass both:

```bash
python -m arena_clients.package --agent-id team-nemo --agent-name "Team Nemo" \
  --entrypoint examples/crewai/agent.py \
  --include examples/crewai/arena_tools.py \
  --lockfile requirements/requirements-all.lock
```

The packager copies only the shared runtime (`arena_clients/`, the strategy
files) and your entrypoint. Add every other file your entrypoint imports with
`--include`, and pass `--lockfile` when it needs packages outside the base lock.
The packager rejects a package whose code imports a local file you did not
include, and names the `--include` to add. It cannot see missing pip packages:
those must be in the lockfile, or the organizer's build fails.

The repository root still has `team-nemo-submission.tar.gz` and
`team-nemo-submission.tar.gz.sha256`. The archive unpacks to a
`gauntlet-submission/` directory with `submission.json` at its root.

## Inside the tarball

When the organizer extracts the archive, the submission tree looks like this:

```text
gauntlet-submission/
├── submission.json
├── requirements.lock
├── arena_clients/
├── base_strategy.py
├── model_selector.py
├── my_strategy.py
└── examples/python_simple/agent.py   # or your declared entrypoint
```

Include every Python module and data file the entrypoint imports at
runtime. The default command copies the starter-kit runtime used by
Python Simple. Add extra files with `--include`.

A worked `submission.json` for that default layout is in
[`examples/python_simple/reference_submission/submission.json`](../examples/python_simple/reference_submission/submission.json).

## `submission.json`

The field contract is [`arena_clients/submission.schema.json`](../arena_clients/submission.schema.json).
The packager writes this file, including `metadata.agentId` and
`metadata.agentName`. Required shape:

```json
{
  "schemaVersion": 1,
  "kind": "GauntletAgentSubmission",
  "metadata": {
    "agentId": "team-nemo",
    "agentName": "Team Nemo"
  },
  "runtime": {
    "language": "python",
    "pythonVersion": "3.12",
    "workingDirectory": ".",
    "entrypoint": ["python", "examples/python_simple/agent.py"]
  },
  "dependencies": {
    "lockfile": "requirements.lock"
  },
  "artifact": {
    "files": [
      "examples/python_simple/agent.py",
      "requirements.lock"
    ]
  }
}
```

Rules the organizer enforces:

- `schemaVersion` is `1` and `kind` is `GauntletAgentSubmission`.
- Runtime is Python 3.12. `workingDirectory` is `"."`.
- `runtime.entrypoint` is an argv array. The first item is `python` or
  `python3`. The second item is a file listed in `artifact.files`. No shell
  script, shell operators, or absolute path.
- `artifact.files` lists every regular file in the directory except
  `submission.json`. Paths are normalized and repository-relative. Do not
  list directories. Do not leave undeclared files.

## Dependency lock

The package must install without downloading floating versions:

```bash
python3.12 -m pip install --require-hashes -r requirements.lock
```

The packager copies `requirements/requirements.lock` to `requirements.lock`
inside the submission. If you added dependencies, add them to
`requirements.in` and regenerate the lock with Python 3.12:

```bash
python -m pip install uv
uv pip compile requirements.in --python-version 3.12 --generate-hashes \
  --output-file requirements/requirements.lock
```

For a CrewAI agent, regenerate the full lock the same way from
`requirements-all.in` into `requirements/requirements-all.lock` (it includes
`requirements.in`).

Then install from it once (`pip install -r requirements.txt`) and run your
agent before packaging, to check that the lock is complete.

The agent must not install packages while it runs.

## What must never be included

Leave these out of the **archive** (inside `gauntlet-submission/`):

- `.env` and `.env.*`
- API keys, tokens, cookies, private keys, and certificates
- `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/`, and tool caches
- Logs, receipts, previous challenge outputs, generated images, and local databases
- Tests, documentation, editor settings, and unused framework examples
- A README, LICENSE, or `.gitignore` added on top of the packaged tree
- Symlinks
- A participant Dockerfile — the organizer builds the image from its own recipe

Those README, LICENSE, and `.gitignore` rules apply to files **inside the
tarball**. Organizers download only the named tarball and checksum pair from
the submission repository.

Never print secret values while packaging.

## Package and self-check

From the starter-kit root, with `.env` loaded so `$AGENT_ID` is set:

```bash
set -a; . ./.env; set +a
python -m arena_clients.package --agent-id "$AGENT_ID" --agent-name "Your Team Name"
python -m arena_clients.package --check dist/gauntlet-submission
```

Package with the same `AGENT_ID` you use in `.env`; it becomes your agent's
identity at the event. Packaging again stops with `output directory already
exists`; add `--force` to replace the earlier build.

That command writes `dist/gauntlet-submission/`,
`dist/<agent-id>-submission.tar.gz`, and
`dist/<agent-id>-submission.tar.gz.sha256`. It fails before writing
when the directory would be rejected: schema, normalized paths, missing
or undeclared files, symlinks, per-file and total size limits,
credential-shaped names, and secret heuristics.

Useful flags:

| Flag | Purpose |
|---|---|
| `--agent-id ID` | Required. Your team's `AGENT_ID` from `.env` (lowercase letters, digits, hyphens). |
| `--agent-name NAME` | Required. Leaderboard display name. |
| `--entrypoint PATH` | Python file to launch (default: `examples/python_simple/agent.py`) |
| `--entrypoint-arg ARG` | Extra argv item after the script; repeatable |
| `--include PATH` | Extra source-relative file to copy; repeatable |
| `--lockfile PATH` | Hash-locked requirements file to copy |
| `--output DIR` | Destination for `gauntlet-submission/` and the tarball (default: `dist/`) |
| `--force` | Replace an existing `gauntlet-submission/` directory |

The packaged agent reads `ARENA_SERVER`, `ARENA_API_KEY`, and `AGENT_ID` from
the process environment, which the organizer's runner sets. It must work when
no `.env` file is present.

## Handoff

Create a **new public repository** (not a fork of this kit), for example
`https://github.com/<your-org>/<agent-id>-submission.git`. The organizer
downloads it without logging in, from its default branch, so a private
repository cannot be fetched and the submission is rejected.

Push `dist/<agent-id>-submission.tar.gz` and
`dist/<agent-id>-submission.tar.gz.sha256` to the repository root. Keep
the filenames exactly as the packager wrote them. Send the organizer that
repository URL.

Do not push the unpacked `dist/gauntlet-submission/` tree. Do not send a
lone `my_strategy.py`. Do not use the host's auto-generated source zip. Do
not send API keys; the organizer supplies runtime configuration.
