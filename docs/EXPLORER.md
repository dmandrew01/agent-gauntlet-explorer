# Gauntlet Explorer

An Agent Gauntlet competitor built on starter kit v1.2.3. It enters an MCP
server with no prior map, inventories every tool, resource and prompt,
decodes hidden data, follows leads to unlisted tools, solves the challenge,
and submits once before the deadline.

Entrypoint: `python examples/explorer/agent.py`

## What the gauntlet will ask (prediction)

The kit fixes the contract. The event blurb adds the black box.

1. **Contract.** One HTTPS origin. REST at `/api` (register, thought, draft,
   submit), MCP over SSE at `/sse`, an OpenAI-style LLM proxy at `/proxy`.
   Register once, wait for GO, fetch the challenge, submit once. A second,
   different answer gets HTTP 409.
2. **Text puzzles.** Logic, reasoning, retrieval, structured output. Practice
   limits run 60 to 150 seconds. An LLM judge checks the answer and its format.
3. **Image puzzles.** Edit a given image or generate one. An image judge
   scores accuracy, preservation and compliance.
4. **Black-box extras.** Tools beyond `arena.*`, tool names hidden in clues or
   descriptions, encoded or lightly encrypted payloads, resources and prompts,
   tool lists that change mid-run, and possibly server requests for sampling
   or elicitation.
5. **Scoring.** Quality 70%, speed 15%, tool coverage 10% (unique tools, capped
   at 3), tokens 5%. Fewer than one broadcast thought costs 10% of the final
   score. On Practice, skipping a `required_tools` entry zeroes quality.

## How the agent covers each one

| Risk | Answer in code |
|---|---|
| Missed thought penalty | First thought goes out right after registration (`runner.Thoughts`) |
| Slow start after GO | MCP connected, inventory taken and model roster fetched during the lobby; GO polled every 0.4 to 0.8 s |
| Unknown tools | `explorer.BlackBox.inventory` lists tools (paginated), resources, templates, prompts and server instructions |
| Calling something dangerous | Token-based classifier; `submit`, `reset`, `register`, `delete`, `leaderboard` and destructive-annotated tools never run from probes or the LLM loop |
| Hidden tools | Every result is mined for tool names in call phrases ("use vault.open"), backticks and dotted names; JSON field names are ignored; unlisted names get probed with any key found |
| Encrypted data | `decoders.auto_decode`: beam search over base64/32/85, hex, binary, decimal, morse, URL, rot13, Caesar, Atbash, rot47, reverse, rail fence, single-byte XOR, zlib/gzip/bz2/lzma; Vigenere and repeating XOR with keys found elsewhere |
| Data hidden in images | `pngtools.inspect_image`: PNG text chunks, bytes after IEND, LSB payloads, JPEG strings; stdlib only |
| Server asks the client for things | Sampling requests answered through the proxy LLM; elicitation answered with schema-valid values; logs and notifications land in the notebook |
| Tool list changes | A re-list follows any call that might unlock something, and any `tools/list_changed` notice (some SDK versions swallow the notice) |
| Required tools | Read from `required_tools` plus any tool name mentioned in the challenge; called in parallel before solving |
| Tool coverage | Clue list, clue get and challenge already make 3; a cheap timer call tops up when needed |
| Model choice | `models.rank_text_models`: quality priors per family, speed weight by time limit, reasoning bonus on hard puzzles, measured stats from calibration when present |
| Model without tool calling | LLM loop switches to a JSON action protocol on the first rejection |
| Format mistakes | `answer.infer_format` reads the rules (JSON keys, single word, integer, case, yes/no, choices); `answer.enforce` repairs mechanically; one LLM repair call only when problems remain |
| Wrong answer when a checker exists | A non-final checker tool (`verify_*`, `check_*`) gets the candidate first; a rejection sends the loop back to work |
| Crash or slow model | The solver runs under a hard deadline; on timeout or exception the best draft (or a fallback) still gets submitted |
| Lost submit response | Same answer retried; 409 treated as already submitted |

Routing per run:

- **Direct** (most logic puzzles): the top-ranked model and a fast partner
  from another family answer in parallel with clues, decoded findings and
  tool evidence. The primary's answer goes in the moment it arrives (Practice:
  right 3 of 3, and each waited second costs about 1.7 speed points on a 60 s
  puzzle). A slow primary gets a bounded wait (15 s on 60 s puzzles) once the
  partner has answered, then the partner's answer goes in. When both arrive
  together and disagree, a third model breaks the tie.
- **Agentic**: when the challenge names extra tools, points at hidden ones,
  asks for exploration or fresh data, or the direct call fails. Probes leads
  for up to 20% of the remaining time, then runs the reason-act loop.
- **Unknown box** (no `arena.*` tools at all): find a challenge-like tool,
  explore, solve, submit through the discovered submit tool.

## Run it

Practice (fill `.env` first: `ARENA_SERVER`, `ARENA_API_KEY`, `AGENT_ID`):

```bash
python -m arena_clients.doctor
python examples/explorer/agent.py
```

Several Practice runs with a score table:

```bash
python tools/practice_loop.py --runs 5
```

Local mock arena (no keys, no network):

```bash
python tools/run_mock_suite.py --deadline   # 8 scenarios, prints PASS/FAIL
python -m unittest tests.test_gauntlet_explorer
```

Model calibration on Practice (writes `gauntlet/model_stats.json`, which the
ranker and the packager pick up):

```bash
python tools/calibrate_models.py
```

Package and self-check:

```bash
python tools/package_explorer.py --agent-name "AHGroup" --force
# publish dist/<agent-id>-submission.tar.gz and .sha256 to a new public repo
```

## Knobs (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `GAUNTLET_MODEL_PREFERENCE` | empty | Comma list of model-name fragments ranked first for text |
| `GAUNTLET_IMAGE_MODEL_PREFERENCE` | empty | Same for image models |
| `GAUNTLET_MODEL_STATS` | `gauntlet/model_stats.json` | Calibration file path |
| `GAUNTLET_FORCE_AGENTIC` | off | Always run the exploration loop |
| `GAUNTLET_PROBE` | on | Set `0` to skip lead probing before the loop |
| `GAUNTLET_MAX_STEPS` | 14 | Reason-act step cap |
| `GAUNTLET_ALLOW_PYTHON` | on | Local Python tool for the LLM (scrubbed env, 8 s cap) |
| `GAUNTLET_TOOL_RESULT_CHARS` | 2500 | Tool output size fed back to the LLM |
| `GAUNTLET_LOBBY_WAIT_S` | 3600 | Max lobby wait |
| `GAUNTLET_HEDGE` | on | Race two models from different families on direct solves; a third breaks ties |
| `GAUNTLET_CONFIRM_WAIT_S` | 0 | Seconds to wait for the partner after the primary answers (0 = submit at once) |
| `GAUNTLET_IMAGE_ANALYZE` | off | Describe the input image with image_analyze before editing |
| `GAUNTLET_IMAGE_HEDGE` | on | Run the top two image models at once |
| `GAUNTLET_IMAGE_PREFER_WAIT_S` | 3 | How long the first-ranked image model may trail the second |
| `GAUNTLET_KEEP_WARM` | on | Keep the REST connection warm until submit, and pre-open the LLM proxy connection |
| `GAUNTLET_THOUGHTS` | private | `verbose` adds answers, models and tool output to broadcast thoughts (other Practice teams see them) |
| `GAUNTLET_BLACKBOX_TIME_S` | 300 | Budget when the server has no timer |

## Verified, and not yet

Practice round 1 (before racing): image edit 86.75 and 86.05, image generate
90.0, logic hard 98.75, logic easy 34.1 (slow, wrong model on a 60 s limit).
Practice round 2 (racing + calibration): logic medium 97.55, logic easy 95.75,
logic hard 98.15, image generate 83.9 (constraints 70: text in a no-text
poster), image edit not scored (submit failed after the edit). Round 3 fixes:
no confirmation wait, no pre-submit network calls, explicit no-text and
preserve-everything image prompts, image submit fallback (REST, then the next
image model, then the input image).
Practice round 3 (run on round 2 code; the pull had failed): logic medium
judged 80 for a "Team " prefix the previous judge accepted; every image run
lost about 20 s to a silent first-call failure of the preferred image model.
Round 4 fixes: strip generic label prefixes shared by all list items, bare-name
prompt hint, run the top two image models in parallel, retry without agent_id
only on real argument-validation errors.
Practice round 4: all 5 accepted, average 91.0. Logic medium 98.3, logic easy
97.55 (submits at 7.2 s), image edit 87.85, image generate 82.1 and 89.15.
gemini-3.1-flash-lite-image returned in ~7 s every time and judged level with
pro; pro never finished, and the 15 s preference window wasted ~15 s per image
run. Round 5: fast image models first (flash-lite, flash, mini, then pro), 3 s
preference window, warm REST and proxy connections, stronger no-text and
minimal-edit prompts.

Verified here: all 12 mock scenarios pass (easy logic, ordering puzzle,
double-encoded clue, hidden vault behind a hex note, image edit, image
generate, slow LLM against a 30 s limit, model without tool calling, slow
primary model, wrong primary model settled by a tie-breaker, oversized image
retried with the next model, preferred image model returning no image); 21 unit tests; kit tests for model selection and
packaging; packager `--check` passes; the unpacked tarball runs with a clean
environment and no `.env`.

Not verified: a live run on Practice, real model behavior, and mcp 1.29 (this
build machine had mcp 2.3, no PyPI access, and a network policy that blocks
the Practice host). The code reads every MCP field
through `gauntlet/compat.py`, which handles both versions. Do one Practice run
per modality before freezing.

## Files added or changed on top of the kit

```text
gauntlet/                  agent package (11 modules, stdlib + kit deps only)
examples/explorer/agent.py entrypoint
my_strategy.py             model ranking and prompts (examples use it too)
tools/package_explorer.py  packager wrapper (includes every gauntlet file)
tools/calibrate_models.py  roster latency and accuracy probe
tools/run_mock_suite.py    end-to-end mock runs
tools/practice_loop.py     repeated Practice runs with a score table
tools/show_run.py          paste-ready report of one run (models, timings, score)
SUBMIT.md                  step-by-step run and submission guide
mock_arena/server.py       local REST + MCP + proxy black box (dev only)
tests/test_gauntlet_explorer.py
docs/EXPLORER.md
```

If the organizers ship a newer kit, copy these paths onto it, rerun the
mock suite and the unit tests, then one Practice run.
