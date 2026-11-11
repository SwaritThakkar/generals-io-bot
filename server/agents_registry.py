"""Loading agents at runtime, including Python files uploaded from the page.

An uploaded file is executed with the project root on `sys.path`, in this
interpreter, so it gets the full environment -- JAX, your checkpoints, whatever
you import. That is the point: a browser cannot run JAX, and this can.

It is also, unavoidably, `exec` on a file you chose. The server binds to
localhost and refuses uploads on any other interface, which makes this exactly as
dangerous as `python youragent.py` and no more.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Names we try, in order, when the file does not say which function to use.
CONVENTIONAL_NAMES = ("act", "agent", "policy", "move", "get_action", "step")


@dataclass
class Agent:
    id: str
    label: str
    kind: str                       # human | builtin | uploaded
    fn: Callable[..., Any] | None = None
    source_name: str | None = None  # original filename, for display
    func_name: str | None = None
    functions: list[str] = field(default_factory=list)  # other callables in the file
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "sourceName": self.source_name,
            "funcName": self.func_name,
            "functions": self.functions,
            "error": self.error,
        }


class AgentLoadError(Exception):
    """Raised with a message meant to be read by a human in the browser."""


def _public_callables(module) -> list[str]:
    out = []
    for name, value in vars(module).items():
        if name.startswith("_") or not callable(value):
            continue
        # Skip things the file merely imported; we want what it defined.
        if getattr(value, "__module__", None) != module.__name__:
            continue
        out.append(name)
    return out


def load_from_source(source: str, filename: str, func: str | None = None) -> Agent:
    """Execute `source` and pick out the agent callable.

    Raises `AgentLoadError` with the file's own traceback when the module fails
    to import, so you can read your own error in the page instead of the console.
    """
    stem = pathlib.Path(filename).stem or "agent"
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem) or "agent"
    directory = pathlib.Path(tempfile.mkdtemp(prefix="genmod_agent_"))
    path = directory / f"{safe}.py"
    path.write_text(source)

    module_name = f"genmod_uploaded_{safe}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AgentLoadError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise AgentLoadError(
            f"{filename} raised while importing:\n\n{traceback.format_exc(limit=8)}"
        )

    candidates = _public_callables(module)
    chosen = func
    if chosen is None:
        for name in CONVENTIONAL_NAMES:
            if name in candidates:
                chosen = name
                break
    if chosen is None and len(candidates) == 1:
        chosen = candidates[0]
    if chosen is None:
        raise AgentLoadError(
            f"{filename} defines {len(candidates)} functions and none is called "
            f"'act'. Rename one to act(state, player), or pick one: "
            f"{', '.join(candidates) or '(none found)'}"
        )
    fn = getattr(module, chosen, None)
    if not callable(fn):
        raise AgentLoadError(f"{filename} has no callable named {chosen!r}")

    return Agent(
        id=f"upload:{uuid.uuid4().hex[:8]}",
        label=f"{filename}:{chosen}",
        kind="uploaded",
        fn=fn,
        source_name=filename,
        func_name=chosen,
        functions=candidates,
    )


def validate(agent: Agent, sample_state, player: int) -> None:
    """Call the agent once so a broken one fails at upload, not mid-game."""
    from server.bot_protocol import normalise

    try:
        result = agent.fn(sample_state, player)  # type: ignore[misc]
    except Exception:
        raise AgentLoadError(
            f"{agent.label} raised when called as {agent.func_name}(state, player):"
            f"\n\n{traceback.format_exc(limit=8)}"
        )
    try:
        actions = normalise(result, sample_state)
    except Exception:
        raise AgentLoadError(
            f"{agent.label} returned {result!r}, which is not an action. Return a "
            "flat action index, a (row, col, dir, half) tuple, or a list of those."
        )
    legal = set(sample_state.legal_actions(player))
    if actions and actions[0] != 0 and actions[0] not in legal:
        # Not fatal: a first move can legitimately be stale by the time it is played.
        agent.error = (
            f"warning: the first move it returned ({actions[0]}) is not legal on the "
            "opening board. It will pass on ticks where its move is invalid."
        )


def builtin_agents() -> list[Agent]:
    from genmod.bots import BOTS
    from server.bot_protocol import load_agent

    out = [Agent(id="human", label="You (keyboard)", kind="human")]
    for name in sorted(BOTS):
        out.append(
            Agent(id=f"builtin:{name}", label=f"{name} (built-in)", kind="builtin",
                  fn=load_agent(name), func_name=name)
        )
    return out
