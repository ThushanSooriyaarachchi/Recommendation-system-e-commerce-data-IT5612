"""Run every Streamlit page through `AppTest` and report exceptions/warnings."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def run_page(label: str, script: Path) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    try:
        at = AppTest.from_file(str(script), default_timeout=120)
        at.run()
    except Exception as exc:  # noqa: BLE001
        return False, [f"FATAL: {exc!r}"]

    if at.exception:
        for e in at.exception:
            msgs.append(f"EXCEPTION: {e.value}")
    if at.error:
        for e in at.error:
            msgs.append(f"ERROR:     {e.value}")
    if at.warning:
        for w in at.warning:
            msgs.append(f"WARNING:   {w.value}")
    return (len(msgs) == 0), msgs


def main() -> int:
    pages: list[tuple[str, Path]] = [
        ("Home", ROOT / "dashboard" / "app.py"),
    ] + [
        (p.stem, p) for p in sorted((ROOT / "dashboard" / "pages").glob("*.py"))
    ]

    overall_ok = True
    for label, script in pages:
        ok, msgs = run_page(label, script)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label} ({script.name})")
        for m in msgs:
            print(f"    {m}")
        if not ok:
            overall_ok = False
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
