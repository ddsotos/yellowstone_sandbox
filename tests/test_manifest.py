import hashlib
from pathlib import Path


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


def test_manifest_hashes_match_bundle_files() -> None:
    # manifestが全対象ファイルを重複なく列挙しSHA-256が一致することを確認する。
    root = Path(__file__).resolve().parents[1]
    lines = (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()

    assert lines
    declared_paths: set[str] = set()
    for line in lines:
        digest, separator, relative_path = line.partition("  ")
        assert separator == "  "
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
        assert relative_path
        assert relative_path == Path(relative_path).as_posix()
        path = root / relative_path
        assert path.is_file()
        assert path.resolve().is_relative_to(root.resolve())
        assert relative_path not in declared_paths
        declared_paths.add(relative_path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    discovered_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not (EXCLUDED_PARTS & set(path.parts))
        and path.is_file()
        and path != root / "MANIFEST.sha256"
        and path.suffix != ".pt"
    }
    assert declared_paths == discovered_paths
