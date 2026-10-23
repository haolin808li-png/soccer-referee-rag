import json

import pytest

from referee_ai.io import atomic_write_json, atomic_write_jsonl, read_jsonl


def test_atomic_json_and_jsonl(tmp_path):
    json_path = tmp_path / "nested" / "value.json"
    jsonl_path = tmp_path / "rows.jsonl"
    atomic_write_json(json_path, {"中文": 1})
    atomic_write_jsonl(jsonl_path, [{"id": 1}, {"id": 2}])
    assert json.loads(json_path.read_text()) == {"中文": 1}
    assert list(read_jsonl(jsonl_path)) == [{"id": 1}, {"id": 2}]


def test_jsonl_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": 1}\nnot-json\n')
    with pytest.raises(ValueError, match=":2"):
        list(read_jsonl(path))
