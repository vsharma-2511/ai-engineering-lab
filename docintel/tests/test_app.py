"""Smoke test: the Streamlit app renders (no model calls, temp project).

Run from the project folder:  pytest
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from docintel import config  # noqa: E402

APP = Path(__file__).parents[1] / "app.py"


@pytest.fixture
def empty_project(tmp_path, monkeypatch):
    for name, value in {
        "PROJECT_ROOT": tmp_path,
        "DOCUMENTS_DIR": tmp_path / "documents",
        "ARCHIVE_DIR": tmp_path / "archive",
        "DB_PATH": tmp_path / "data" / "registry.db",
        "PARSED_DIR": tmp_path / "data" / "parsed",
        "UPLOADS_DIR": tmp_path / "data" / "uploads",
    }.items():
        monkeypatch.setattr(config, name, value)
    # A provider that fails fast without a key, instead of calling Ollama.
    monkeypatch.setattr(config, "QA_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_app_renders_and_explains_what_is_missing(empty_project):
    app = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not app.exception
    assert any("OPENAI_API_KEY" in error.value for error in app.sidebar.error)
    assert app.chat_input[0].disabled  # no working model yet
    assert any("working model" in info.value for info in app.info)

