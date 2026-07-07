import os as _os


def _parse_env_file(path):
    """Load key=value pairs from *path* into os.environ, skipping keys already set."""
    try:
        with open(path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _, _v = _line.partition('=')
                    _k, _v = _k.strip(), _v.strip()
                    if _k and _k not in _os.environ:
                        _os.environ[_k] = _v
        return True
    except OSError:
        return False


# 1. Use env var if already exported before launching Isaac Sim (e.g. in ~/.bashrc).
_root = _os.environ.get("KITTING_PROJECT_ROOT", "")

# 2. If not set, walk $HOME up to 3 levels deep to find a .env containing it.
#    Isaac Sim's Script Editor copies scripts to /tmp/carb.*/ so __file__ is
#    unreliable — project root cannot be derived from the script path.
if not _root:
    _home = _os.path.expanduser("~")
    for _dp, _dns, _fns in _os.walk(_home):
        _rel = _os.path.relpath(_dp, _home)
        if _rel != '.' and _rel.count(_os.sep) >= 3:
            _dns[:] = []
            continue
        _dns[:] = [d for d in _dns if not d.startswith('.')]
        if ".env" in _fns:
            try:
                _ev = _os.path.join(_dp, ".env")
                with open(_ev) as _ef:
                    _content = _ef.read()
                if "KITTING_PROJECT_ROOT" in _content:
                    _parse_env_file(_ev)
                    _root = _os.environ.get("KITTING_PROJECT_ROOT", "")
                    if _root:
                        break
            except OSError:
                pass

if not _root:
    raise RuntimeError(
        "KITTING_PROJECT_ROOT is not set and no .env containing it was found.\n"
        "Add this line to ~/.bashrc, then restart your terminal and re-launch Isaac Sim:\n"
        "  export KITTING_PROJECT_ROOT=/path/to/robot_in_air"
    )

exec(open(_os.path.join(_root, "generative_kitting", "isaac_sim_bridge.py")).read())
