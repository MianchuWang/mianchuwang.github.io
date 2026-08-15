"""Runs the registered cases and leaves a JSON summary in `RESULT`."""

import json
import traceback

_results = []

for _name, _fn in CASES:
    try:
        _fn()
        _results.append({"name": _name, "ok": True, "detail": ""})
    except AssertionError as exc:
        _results.append({"name": _name, "ok": False, "detail": str(exc)})
    except Exception as exc:  # an error raised inside the user's function
        _frames = [
            f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "your_code.py"
        ]
        _where = ""
        if _frames:
            _f = _frames[-1]
            _where = f"\n    your_code.py, line {_f.lineno}, in {_f.name}:  {(_f.line or '').strip()}"
        _results.append(
            {
                "name": _name,
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}{_where}",
            }
        )

RESULT = json.dumps(_results)
