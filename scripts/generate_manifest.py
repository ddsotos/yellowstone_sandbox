"""Generate the SHA-256 manifest for tracked bundle content."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {
    ".venv",
    ".pytest_cache",
    ".test-tmp-fast-enum",
    "__pycache__",
    "data",
    "logs",
    "models",
    "results",
}


def main() -> None:
    entries: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if (
            EXCLUDED_PARTS & set(path.parts)
            or not path.is_file()
            or path == MANIFEST
            or path.suffix == ".pt"
        ):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    temporary = MANIFEST.with_suffix(f".sha256.{os.getpid()}.tmp")
    temporary.write_text("\n".join(entries) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)


if __name__ == "__main__":
    main()
