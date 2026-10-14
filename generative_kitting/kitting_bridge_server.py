"""Start the Isaac Sim bridge from Isaac Sim's Script Editor (or MCP execute_script).

    exec(open("/path/to/robot_in_air/generative_kitting/kitting_bridge_server.py").read())

Finds the repo root (the folder holding generative_kitting/isaac_sim_bridge.py)
without any machine-specific setting, then exec()s the bridge. Tried in order:
  1. KITTING_PROJECT_ROOT as a variable in the calling namespace or the environment
  2. __file__, when it points into the repo (the Script Editor runs a /tmp copy)
  3. the open USD stage's folder and its parents (the scene lives in the repo)
  4. the current directory and its parents
  5. a search of $HOME, 6 levels deep, skipping hidden folders
Every candidate must contain the marker file, so a stale value is skipped.
"""
import os as _os

_MARKER = _os.path.join("generative_kitting", "isaac_sim_bridge.py")


def _repo_root_from(start, levels=8):
    """``start`` or the first parent that contains the marker, else None."""
    if not start:
        return None
    d = _os.path.abspath(_os.path.expanduser(start))
    if _os.path.isfile(d):
        d = _os.path.dirname(d)
    for _ in range(levels):
        if _os.path.isfile(_os.path.join(d, _MARKER)):
            return d
        parent = _os.path.dirname(d)
        if parent == d:
            return None
        d = parent
    return None


def _stage_dir():
    try:
        import omni.usd
        url = omni.usd.get_context().get_stage_url() or ""
    except Exception:
        return None
    for prefix in ("file:///", "file://", "file:"):
        if url.startswith(prefix):
            url = "/" + url[len(prefix):].lstrip("/") if _os.name != "nt" else url[len(prefix):]
            break
    return _os.path.dirname(url) if url else None


def _search_home(max_depth=6):
    home = _os.path.expanduser("~")
    for dirpath, dirnames, _ in _os.walk(home):
        rel = _os.path.relpath(dirpath, home)
        if rel != "." and rel.count(_os.sep) >= max_depth - 1:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if _os.path.isfile(_os.path.join(dirpath, _MARKER)):
            return dirpath
    return None


def _find_repo_root():
    explicit = globals().get("KITTING_PROJECT_ROOT") or _os.environ.get("KITTING_PROJECT_ROOT", "")
    for candidate in (explicit, globals().get("__file__"), _stage_dir(), _os.getcwd()):
        root = _repo_root_from(candidate)
        if root:
            return root
    return _search_home()


_root = _find_repo_root()
if not _root:
    raise RuntimeError(
        "Could not find the robot_in_air repo (no generative_kitting/isaac_sim_bridge.py "
        "near the open stage, the current directory or $HOME). Open the project's USD "
        "scene first, or set the path before running this file:\n"
        '  KITTING_PROJECT_ROOT = "/path/to/robot_in_air"')
_os.environ["KITTING_PROJECT_ROOT"] = _root

# The bridge locates .env and the robot config from its own __file__.
__file__ = _os.path.join(_root, _MARKER)
exec(compile(open(__file__, encoding="utf-8").read(), __file__, "exec"))
