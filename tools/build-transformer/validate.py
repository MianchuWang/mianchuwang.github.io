#!/usr/bin/env python3
"""Dev check: run every component's reference solution against its own tests.

The browser runs exactly these files through Pyodide; this script runs them
with the local interpreter so a broken test is caught before it ships.

    python3 tools/build-transformer/validate.py          # all components
    python3 tools/build-transformer/validate.py rope mha # a subset

Needs numpy and einops in the current interpreter.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"


def read(path):
    return path.read_text(encoding="utf-8")


def run_component(comp, code_file):
    """Exec harness + preamble + code + tests + runner in a fresh namespace."""
    ns = {}
    exec(compile(read(DATA / "harness.py"), "harness.py", "exec"), ns)
    run_source = ns["run_source"]
    cdir = DATA / comp["id"]
    if comp.get("preamble"):
        run_source(read(cdir / "preamble.py"), "preamble.py")
    run_source(read(cdir / code_file), "your_code.py")
    run_source(read(cdir / "tests.py"), "tests.py")
    run_source(read(DATA / "runner.py"), "runner.py")
    return json.loads(ns["RESULT"])


def main(argv):
    components = json.loads(read(DATA / "components.json"))["components"]
    wanted = set(argv[1:])
    if wanted:
        components = [c for c in components if c["id"] in wanted]
        missing = wanted - {c["id"] for c in components}
        if missing:
            print(f"unknown component(s): {', '.join(sorted(missing))}")
            return 2

    failures = 0
    for comp in components:
        cdir = DATA / comp["id"]
        for name in ["brief.html", "starter.py", "solution.py", "tests.py"]:
            if not (cdir / name).exists():
                print(f"FAIL {comp['id']}: missing {name}")
                failures += 1

        try:
            results = run_component(comp, "solution.py")
        except Exception as exc:
            print(f"FAIL {comp['id']}: solution did not run — {type(exc).__name__}: {exc}")
            failures += 1
            continue

        bad = [r for r in results if not r["ok"]]
        if not results:
            print(f"FAIL {comp['id']}: no test cases registered")
            failures += 1
        elif bad:
            print(f"FAIL {comp['id']}: {len(bad)}/{len(results)} cases failed")
            for r in bad:
                print(f"       - {r['name']}: {r['detail'].splitlines()[0]}")
            failures += 1
        else:
            # The starter must fail, otherwise the exercise tests nothing.
            try:
                starter = run_component(comp, "starter.py")
                starter_ok = all(r["ok"] for r in starter) and bool(starter)
            except Exception:
                starter_ok = False
            if starter_ok:
                print(f"FAIL {comp['id']}: the starter code already passes")
                failures += 1
            else:
                print(f"ok   {comp['id']:<18} {len(results)} cases")

    print(f"\n{len(components) - failures}/{len(components)} components ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
