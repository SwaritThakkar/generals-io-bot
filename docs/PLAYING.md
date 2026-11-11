# Playing the game

Three ways to play the perfect-information variant: a hosted web app you can
send to a friend, a local server that plays on the JAX engine, and a bot
protocol that lets any Python agent take a seat in either.

```
viewer.py     native pygame window — watch, play, and rewind (JAX engine)
server/       local web server + the genmod-bot/1 protocol (JAX engine)
legacy/web/   Next.js app for Vercel — TypeScript engine, Redis-backed matches
legacy/shared/ board.mjs, the canvas renderer both web front ends use
legacy/part1/tools/export_golden.py — generates the cross-implementation test vectors
```

---

## Two engines, one rule set

The hosted app cannot keep a JAX process alive per match, so its rules are a
TypeScript mirror in [web/lib/engine.ts](../legacy/web/lib/engine.ts). To stop the two
drifting, `python legacy/part1/tools/export_golden.py` runs games in JAX and records the
board after every tick; [web/test/golden.test.ts](../legacy/web/test/golden.test.ts)
replays them through the TypeScript engine and asserts equality cell by cell —
17 cases, 605 ticks, covering combat, ties, castle drains, production, general
capture and free-for-all transfers.

**If you change the rules in `genmod/env.py`, change `legacy/web/lib/engine.ts` too,
regenerate the vectors, and run `npm test`.** That test failing means an agent
would be trained against different rules from the ones it is evaluated on.

Map *generation* is not covered by those vectors: the two implementations sample
the same distribution from different PRNGs. `legacy/web/test/mapgen.test.ts` checks the
constraints instead (mountain density, castle count and garrisons, spawn
distance, the guaranteed castle within radius 6).

Locally, `server/local_server.py` runs the JAX engine directly — no TypeScript
in the loop at all.

---

## The pygame window

A native desktop viewer. No browser, no server — just a window.

```bash
# watch two agents fight
python viewer.py --p0 expander --p1 server/agents/example_agent.py:act

# play one yourself
python viewer.py --p0 human --p1 expander

# your trained policy, on the JAX engine
python viewer.py --p0 human --p1 server/agents/jax_agent.py:act --tick 500
```

Agent specs are the same everywhere: `human`, a builtin (`expander`, `random`,
`pass`), `file.py:function`, `module:function`, or an `http(s)://` endpoint.
Options: `--size`, `--tick` (ms; `0` runs flat out), `--seed`, `--min-spawn`,
`--max-ticks`, `--window WxH`.

**Every tick is kept**, so you can rewind and watch a losing exchange again —
which is the reason to use this over the browser when you are debugging an agent.

| key | while spectating |
|---|---|
| `space` | pause / resume |
| `←` `→` | step back / forward through history (`shift` for 10) |
| `home` / `end` | jump to the start / back to live |
| `[` `]` | slower / faster |
| `r` | new game |
| `q` or `esc` | quit |

When you hold a seat the arrows queue moves instead: **click** one of your tiles,
then **arrows / wasd** to queue a path (**shift** sends half), **c** clears it,
**p** pauses.

Agents run on a background thread, so a slow one — a JAX policy compiling on its
first call — never freezes the window.

## Local play

```bash
python server/local_server.py --open
```

Then open http://localhost:8000. The **Players** panel on the right is the whole
interface: pick who sits in each seat, drop a `.py` file to add your own agent,
and press *Start new game*.

**Loading a `.py` from the page is the point of running locally.** The file is
imported into this interpreter, so it gets your real environment — JAX, your
checkpoints, your GPU. A browser has no XLA, which is why the deployed site
cannot do this and this can.

The file is checked at load time, not mid-game: if it fails to import, if `act()`
raises, if it returns something that is not an action, or if it defines several
functions and none is named `act`, you get the message (and your own traceback)
in the page instead of a silent pass.

You can still drive everything from the command line:

```bash
# watch two of your agents, as fast as they will go
python server/local_server.py --p0 my_agent.py:act --p1 other.py:act --tick 0 --open

# score a series, no browser
python server/local_server.py --p0 expander --p1 random --games 100 --headless
```

Options: `--size`, `--tick` (ms, `0` = as fast as possible), `--seed`,
`--min-spawn`, `--max-ticks`, `--host`, `--port`.

An agent spec is one of:

| spec | meaning |
|---|---|
| `human` | you, in the browser |
| `expander`, `random`, `pass` | the scripted bots in `genmod/bots.py` |
| `mymodule:act` | a callable in an importable module |
| `path/to/file.py:act` | a callable in a file |
| `http://host:port/` | a remote `genmod-bot/1` endpoint |

For large-scale evaluation use `genmod.make_rollout` instead — it batches
thousands of games into one XLA graph. `local_server.py` runs one match at a time
and is meant for watching and for modest series.

### A note on running uploaded files

The page executes the Python file you give it, in this process. That is exactly
as dangerous as `python youragent.py` and no more — but it does mean the server
binds to `127.0.0.1` and **refuses uploads on any other interface**. If you pass
`--host 0.0.0.0`, the file picker disappears and `/api/agent` returns 403.

### Controls

Click one of your tiles, then arrow keys (or WASD) queue a path — each key sends
all but one unit. **Shift + direction** sends half. **Q** or **Esc** clears the
queue. Amber arrows are your pending path; the coloured arrows are what each
side committed on the last tick.

---

## Writing an agent

An agent is a function. That is the whole contract:

```python
from server.bot_protocol import BotState, encode

def act(state: BotState, player: int):
    # state.owner / .army / .mountain / .castle / .general are (h, w) numpy arrays.
    # Nothing is hidden — this is the perfect-information variant.
    return encode(row, col, "right", half=False, width=state.w)
```

Return a flat action index, a `(row, col, dir, half)` tuple, or a list of those
to queue a path. `state.legal_actions(player)` gives you every legal index.
See [server/agents/example_agent.py](../server/agents/example_agent.py).

For a trained JAX policy, `state.to_genmod_state(cfg)` hands you a `genmod.State`
you can feed straight to the observation encoder and your network — no weight
export, no ONNX, no second implementation:

```python
def act(state, player):
    cfg = state.env_config()
    st = state.to_genmod_state(cfg)
    obs = genmod.observe(st, cfg)[player]            # (C, H, W) float32
    mask = genmod.legal_action_mask(st, cfg)[player] # (H, W, 9) bool
    return int(my_policy(obs, mask))
```

[server/agents/jax_agent.py](../server/agents/jax_agent.py) is a working version of
that — drop it on the page and watch it play.

**One gotcha worth knowing:** channel 0 of the action mask is *pass*, and it is
legal on every cell. A flat `argmax` over the masked logits will therefore pass
forever unless you exclude it.

### Serving it to the web app

```bash
python server/agents/example_agent.py --port 8801
# or a built-in, as a known-good opponent:
python server/agents/builtin_service.py --bot expander --port 8801
```

That exposes `POST /` taking `{protocol, player, deadlineMs, state}` and
answering `{"actions": [...]}`. A bot that errors, times out or answers with
nonsense simply passes that tick — one flaky endpoint cannot wedge a match.

---

## The web app

```bash
cd web
npm install
npm run dev            # http://localhost:3000
npm test               # engine golden tests, map constraints, bots, local matches
npm run smoke          # drives a real Chrome against a running server
```

`npm run dev` works with no configuration.

### Two very different paths

**Playing a bot runs entirely in your browser tab.** `/local` is a static route:
the engine, the map generator, the opponent and the clock are all client-side, so
a keypress lands on the very next tick and nothing touches the network. On this
deployment a server round trip costs ~300ms plus ~180ms per Redis command, which
is hopeless for a 500ms tick — so bot matches simply do not use the server.

**Playing a friend needs the server**, because two browsers have to agree on one
board. That path is Redis-backed and server-authoritative: your moves queue on
the server, so lag delays them but never drops them.

### Deploying to Vercel

```bash
cd web
npx vercel login          # interactive, once
npx vercel link           # creates the project
npx vercel --prod         # deploy
```

Deploy from **`legacy/web/` as the root directory** — the research code does not need to
ship. `public/board.mjs` is committed, so the build does not depend on `../shared`
being in the checkout.

**Add storage before you rely on it.** Vercel dashboard → Storage → Redis
(Upstash). It injects `KV_REST_API_URL` and `KV_REST_API_TOKEN`; redeploy after
adding it. Without them, every request can land on a different instance and the
in-memory fallback is not shared between them, so a match can vanish between
ticks. The lobby shows a red banner when it detects this, rather than letting you
find out mid-game.

**Opponents.** Two scripted samples — Expander and Rusher — are compiled into the
app, so the *vs computer* mode works on a fresh deployment with nothing
configured. To add your own agent, set `GENMOD_BOTS` to a JSON array of
`{id, name, description, url}`, or just paste an endpoint into the lobby. The URL
has to be reachable *from Vercel*, so to serve an agent off your laptop, tunnel it:

```bash
python server/agents/example_agent.py --port 8801
cloudflared tunnel --url http://localhost:8801
```

`.env.example` documents every variable.

### How the clock works without a server

Vercel functions are short-lived, so nothing can hold a 0.5s timer. For friend
matches the clock is *derived*: a match records when it started, the current tick
is `floor((now - startedAt) / tickMs)`, and any request advances the stored board
up to that tick under a Redis lock. Whoever touches the match next does the work.

The stream in `app/api/match/[id]/stream/route.ts` deliberately ends after 45
seconds and lets the browser's `EventSource` reconnect, staying well inside the
function ceiling. Every message is a full snapshot, so a reconnect needs no
catch-up logic.

### Friend-match latency, honestly

Upstash answers in roughly 180ms per command from the Vercel region, and the
function itself adds ~300ms. `lockedUpdate` pipelines lock+read into one request
and write+unlock into another, and a poll with nothing to advance skips the lock
entirely, which took a request from a flat ~1330ms down to ~490ms idle and
~900ms median. That is still slower than a 500ms tick, so **a friend match
advances in small bursts rather than smoothly.**

If that bothers you, in rough order of effect:

1. Pick the **1.0s tick** in the lobby for friend matches.
2. Put the Redis database in the same region as the deployment.
3. Move friend matches to a realtime service with a persistent connection
   (Ably, Pusher) instead of polled Redis.

None of this affects bot matches, which never touch the network.

### Cost note

Each open stream polls Redis roughly every 200ms, so a four-minute match costs a
few thousand Redis commands. That is fine for a research project and for
occasional games; it is not free-tier-friendly if you leave tabs open for hours.
Raise `GENMOD_POLL_MS`, or raise `tickMs`, if you start bumping the quota.

### The two built-in opponents

Both live in [web/lib/localbots.ts](../legacy/web/lib/localbots.ts) and run inside the app —
no endpoint, no round trip per tick.

| | what it does |
|---|---|
| **Expander** | Takes any adjacent cell it can afford, otherwise marches its biggest stack down a BFS field to the nearest unowned cell. A port of `genmod.bots.expander_bot`, so local and hosted play meet the same baseline. |
| **Rusher** | Expands through most of the production cycle, pulls army back to its general in the last stretch of it, and once the pile can survive the walk and beat the garrison, walks it straight at your general — which, with no fog, it can see from turn one. |

They are deliberately simple heuristics: the floor a trained agent should clear,
and a way to confirm the app works. `legacy/web/test/localbots.test.ts` holds them to it —
legal actions from either seat, beating a random player, and Rusher closing games
out faster than Expander. Anything stronger belongs behind the `genmod-bot/1`
protocol, where it can be a real JAX policy.
