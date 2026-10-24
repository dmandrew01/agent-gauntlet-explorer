# Run, test, and submit the Gauntlet Explorer

Every command below runs from the repository root unless a step says otherwise.
Python 3.12 is required. Submissions are built and run on 3.12.

Team id (`AGENT_ID`): `dmandrew01`. Team name: `AHGroup` (set in
`my_strategy.py` and passed to the packager). Use both unchanged from Practice
through event day; the server rejects a second registration under a different
name.

## Windows with PyCharm (recommended setup)

1. Install Python 3.12 from https://www.python.org/downloads/windows/
   (64-bit installer). Tick "Add python.exe to PATH" on the first screen.
2. Install Git from https://git-scm.com/download/win (defaults are fine).
3. Install PyCharm (free) from https://www.jetbrains.com/pycharm/download/
4. In PyCharm: Clone Repository, choose GitHub, log in as dmandrew01, pick
   `agent-gauntlet-explorer`, Clone.
5. Settings, Project, Python Interpreter, Add Interpreter, Add Local
   Interpreter, Virtualenv, New, base interpreter Python 3.12, location
   `.venv` inside the project. OK.
6. Open the Terminal tab at the bottom (Alt+F12). The prompt starts with
   `(.venv)`. Every command in this guide runs there. Differences from the
   Linux commands: use `copy .env.example .env` instead of `cp`, and skip
   `source .venv/bin/activate` (PyCharm activates it for you).

## 1. Get the code and install (Linux, macOS, or WSL)

```bash
git clone https://github.com/dmandrew01/agent-gauntlet-explorer.git
cd agent-gauntlet-explorer
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` installs the kit's hash-locked set. The explorer adds no
packages, so the lockfile stays the organizer's original.

## 2. Create `.env`

```bash
cp .env.example .env
```

Set these three lines (no port on the server, no quotes):

```text
ARENA_SERVER=https://gauntlet-practice.136.85.160.148.sslip.io
ARENA_API_KEY=AGENT_GAUNTLET_FALL
AGENT_ID=dmandrew01
```

Never commit `.env`. The repository's `.gitignore` already excludes it.

## 3. Check the setup

```bash
python -m arena_clients.doctor
```

Expect `All checks passed` at the end. If a check fails, the doctor prints the fix.

## 4. Optional: calibrate model choice on Practice

```bash
python tools/calibrate_models.py
```

Takes a few minutes. Writes `gauntlet/model_stats.json` with measured latency
and accuracy per model. The ranker and the packager both pick the file up.

## 5. Run on Practice

One run:

```bash
python examples/explorer/agent.py
```

Five runs with a score table (covers text and image puzzles):

```bash
python tools/practice_loop.py
```

What to check in the table:

- `ok` reads `True` on text runs. `False` means a wrong answer or a skipped
  required tool (the notes column names the tool).
- No `thought penalty` note.
- Image runs show a score and no `judge error`. A judge error is the
  organizer's side; run again.

Each run record lands in `outputs/dmandrew01-run-<n>.json` with the
challenge, every tool call, the answer, and the score breakdown.

Other teams share the Practice key and see your runs. The agent keeps answers,
model names, and tool output out of its broadcast thoughts unless you set
`GAUNTLET_THOUGHTS=verbose`.

## 6. Offline checks (no server needed)

```bash
python -m unittest tests.test_gauntlet_explorer
python tools/run_mock_suite.py --deadline
```

The mock suite runs 8 scenarios (logic, ordering, encoded clue, hidden vault,
image edit, image generate, slow model, model without tool calling) and prints
PASS or FAIL for each.

## 7. Package

```bash
python tools/package_explorer.py --agent-name "AHGroup" --force
```

The script reads `AGENT_ID` from `.env`.

Expect two `PASS` lines. Output:

```text
dist/dmandrew01-submission.tar.gz
dist/dmandrew01-submission.tar.gz.sha256
```

The packager uses the kit's own admission checks: schema, declared files,
size limits, no secrets, no `.env`, no symlinks, no local imports left out.

## 8. Publish the submission repository

1. On github.com, create a **new public** repository named
   `dmandrew01-submission`. Do not fork the starter kit. No template.
2. Put exactly the two packager files at its root (works for an empty
   repository and for later updates):

```bash
git clone https://github.com/dmandrew01/dmandrew01-submission.git ../submission
cp dist/dmandrew01-submission.tar.gz dist/dmandrew01-submission.tar.gz.sha256 ../submission/
cd ../submission
git add dmandrew01-submission.tar.gz dmandrew01-submission.tar.gz.sha256
git commit -m "Gauntlet Explorer submission"
git push -u origin HEAD:main
cd -
```

3. Open the repository page in a private browser window. Both files must be
   visible without logging in.

Rules the organizer enforces: one tarball and one checksum only, file names
exactly as the packager wrote them, no unpacked `gauntlet-submission/` tree,
no source zip, no API keys.

## 9. Send the URL

Send the organizer `https://github.com/dmandrew01/dmandrew01-submission`.

## 10. After changes

Edit, rerun steps 5 and 7, then rerun the copy, commit and push lines of
step 8 (skip the clone if `../submission` already exists; run `git pull` there
first). The organizer reads the default branch.

## Event day

Nothing changes in the code. The organizer's runner sets `ARENA_SERVER`,
`ARENA_API_KEY` and `AGENT_ID` itself. The agent waits in the lobby, starts on
GO, and submits once. The organizers also run `doctor --certify` on your
package against their certification server; you do not run it, and it cannot
pass on Practice.
