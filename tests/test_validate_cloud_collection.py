import gzip
from pathlib import Path

import pytest

from yellowstone.validate_cloud_collection import validate_workers


def _worker(path: Path, ids: list[int]) -> None:
    path.mkdir()
    (path / "collection_manifest.json").write_text("{}", encoding="utf-8")
    lines = "\n".join('{"game_id": %d}' % game_id for game_id in ids) + "\n"
    with gzip.open(path / "games.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(lines)


def test_rejects_cross_worker_duplicate_ids(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _worker(a, [1, 2])
    _worker(b, [2, 3])
    with pytest.raises(ValueError, match="duplicate game ids"):
        validate_workers([a, b])
