"""Environment-aware YAML config loader.

Resolves ``${VAR}`` and ``${VAR:-default}`` placeholders inside string
values against ``os.environ`` so that machine-specific values (server
IPs, install paths, secrets) live in the operator's local environment
rather than in the checked-in YAML.

The loader also exposes a ``PROJECT_ROOT`` placeholder that auto-resolves
to the project root (the parent of ``generative_kitting/``), so YAML
files can reference repo-relative paths like
``${PROJECT_ROOT}/AIKIDO.usd`` without anyone having to set an env
var manually.

This module has zero third-party deps beyond PyYAML so it works the
same in the Streamlit process, the main CLI, and inside Isaac Sim.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_PLACEHOLDER = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def project_root() -> Path:
    """Return the repo root (the parent of ``generative_kitting/``).

    Resolved from this file's location so it works regardless of CWD
    or how the script was launched.
    """
    return Path(__file__).resolve().parent.parent.parent


def _ensure_project_root_env() -> None:
    """Inject ``PROJECT_ROOT`` into the environment if the operator
    hasn't set it. Uses forward slashes for cross-platform YAML."""
    os.environ.setdefault("PROJECT_ROOT", project_root().as_posix())


def _load_dotenv_if_present() -> None:
    """Load ``<project_root>/.env`` into ``os.environ`` so values
    written there (OLLAMA_BASE_URL, ISAACSIM_PATH, …) are visible
    to the config interpolator.

    Uses python-dotenv when available, falls back to a tiny inline
    parser otherwise so this works even in stripped-down Python
    environments (Isaac Sim's bundled interpreter, CI runners, …).

    Values already present in ``os.environ`` are NOT overwritten —
    a shell export always wins over the .env file, matching the
    conventional dotenv precedence.
    """
    env_path = project_root() / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    # Minimal fallback parser: KEY=VALUE per line, # comments, no
    # multi-line values, no variable substitution. Matches what
    # .env.example needs.
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _expand_string(value: str) -> str:
    """Replace every ``${VAR}`` / ``${VAR:-default}`` token in *value*
    with its environment value, leaving non-matching text untouched.
    Repeats until no further substitutions happen (allows one level of
    indirection, e.g. a default that references another env var)."""
    seen: set[str] = set()
    for _ in range(8):  # cap to avoid infinite loops on circular refs
        if value in seen:
            break
        seen.add(value)

        def sub(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)
            if var_name in os.environ:
                return os.environ[var_name]
            if default is not None:
                return default
            # No env value and no default — leave the placeholder so
            # the failure is loud at the point of use rather than
            # silently substituting an empty string.
            return match.group(0)

        new_value = _PLACEHOLDER.sub(sub, value)
        if new_value == value:
            return new_value
        value = new_value
    return value


def _expand(node: Any) -> Any:
    if isinstance(node, str):
        return _expand_string(node)
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    return node


def load_config(config_path: str | os.PathLike) -> dict:
    """Load *config_path* as YAML and resolve env-var placeholders.

    Calls ``load_dotenv`` first so values in ``<project_root>/.env``
    are visible to the placeholder expansion. Shell exports take
    precedence over .env entries.

    The caller is responsible for passing an absolute or
    correctly-relative path — this helper does not search for the
    file."""
    _load_dotenv_if_present()
    _ensure_project_root_env()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _expand(raw)
