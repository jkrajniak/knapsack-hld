"""Re-hash every instance file in `instances/` against `MANIFEST.json`.

Exit code 0 iff every entry matches; non-zero with a printed report
otherwise. Used in CI to guarantee archive integrity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from instances import verify_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("instances"))
    args = parser.parse_args()

    ok, errors = verify_manifest(args.archive)
    if ok:
        print(f"OK: archive at {args.archive} matches MANIFEST.json")
        return 0
    print(f"FAIL: {len(errors)} integrity error(s) in {args.archive}", file=sys.stderr)
    for line in errors:
        print(f"  - {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
