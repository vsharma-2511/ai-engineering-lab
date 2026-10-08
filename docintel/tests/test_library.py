"""Document library and chat-session lifecycle, on real (tiny) PDFs.

Everything runs in a temporary project folder: no LLM, no embedding
model download.  Run from the project folder:  pytest
"""
from datetime import datetime, timedelta, timezone
import sqlite3

import fitz  # PyMuPDF
import numpy as np
import pytest

from docintel import config, sessions
from docintel.chunking.chunk_storage import load_latest_chunks
from docintel.ingestion import watcher
from docintel.ingestion.library import (
    delete_document,
    list_library,
    remove_missing_documents,
)
from docintel.ingestion.registry import get_document_versions, registry_key
from docintel.retrieval.vector_store import load_or_compute_vectors
from docintel.tokens import approximate_token_count


class FakeEncoder:
    name = "fake"

    def encode(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32) / 2


def make_pdf(path, text):
    """A one-page PDF with enough text to pass the parser's quality check."""
    filler = ("This paragraph pads the page so the parser treats it as a "
              "normal text page rather than an empty or scanned one. ") * 3
    document = fitz.open()
    document.new_page().insert_textbox(fitz.Rect(72, 72, 520, 700),
                                       f"{text}\n\n{filler}", fontsize=11)
    document.save(path)
    return path


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project folder wired into config."""
    paths = {
        "PROJECT_ROOT": tmp_path,
        "DOCUMENTS_DIR": tmp_path / "documents",
        "ARCHIVE_DIR": tmp_path / "archive",
        "DATA_DIR": tmp_path / "data",
        "DB_PATH": tmp_path / "data" / "registry.db",
        "PARSED_DIR": tmp_path / "data" / "parsed",
        "UPLOADS_DIR": tmp_path / "data" / "uploads",
    }
    for name, value in paths.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(config, "WATCHER_QUIET_SECONDS", 0)
    for name in ("DOCUMENTS_DIR", "ARCHIVE_DIR"):
        paths[name].mkdir(parents=True)
    watcher.initialize_registry(config.DB_PATH)
    watcher.initialize_chunk_storage(config.DB_PATH)
    watcher.pending.clear()
    watcher.removed_paths.clear()
    return tmp_path


def ingest(path):
    """What the watcher does when a PDF lands in the inbox."""
    watcher.queue_candidate(path)
    watcher.process_ready_candidates(approximate_token_count)


def rows(table):
    with sqlite3.connect(config.DB_PATH) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# --- library ------------------------------------------------------------

def test_processed_pdf_moves_to_archive(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Revenue grew 5%."))

    assert not (config.DOCUMENTS_DIR / "Report.pdf").exists()
    assert (config.ARCHIVE_DIR / "Report.v1.pdf").is_file()
    [record] = get_document_versions(config.DB_PATH, "Report.pdf")
    assert record["status"] == "CHUNKED"
    assert record["archive_path"].endswith("Report.v1.pdf")
    assert load_latest_chunks(config.DB_PATH)  # searchable from the archive
    assert list_library(config.DB_PATH)[0]["location"] == (
        "archive/Report.v1.pdf"
    )


def test_new_version_is_archived_under_its_own_name(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Version one."))
    ingest(make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Version two."))

    assert sorted(p.name for p in config.ARCHIVE_DIR.iterdir()) == [
        "Report.v1.pdf", "Report.v2.pdf",
    ]
    texts = [c["text"] for c in load_latest_chunks(config.DB_PATH)]
    assert any("Version two" in t for t in texts)
    assert not any("Version one" in t for t in texts)


def test_delete_document_removes_everything(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Version one."))
    ingest(make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Version two."))
    load_or_compute_vectors(config.DB_PATH,
                            load_latest_chunks(config.DB_PATH), FakeEncoder())
    assert rows("chunk_vectors") > 0

    summary = delete_document(config.DB_PATH, "Report.pdf")

    assert summary["versions"] == 2
    for table in ("documents", "chunks", "chunk_vectors"):
        assert rows(table) == 0, table
    assert list(config.ARCHIVE_DIR.iterdir()) == []
    assert list(config.PARSED_DIR.iterdir()) == []
    assert delete_document(config.DB_PATH, "Report.pdf") is None


def test_delete_keeps_an_unregistered_new_file_in_the_inbox(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Version one."))
    newer = make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Not seen yet.")

    delete_document(config.DB_PATH, "Report.pdf")

    assert newer.is_file()


def test_deleting_the_archived_pdf_removes_the_document(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Revenue grew."))
    archived = config.ARCHIVE_DIR / "Report.v1.pdf"

    archived.unlink()
    watcher.queue_removal(archived)
    watcher.process_removals()

    assert rows("documents") == 0 and rows("chunks") == 0


def test_archiving_itself_does_not_count_as_deletion(project):
    inbox = make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Revenue grew.")
    ingest(inbox)

    # The move out of the inbox fires a "moved"/"deleted" event.
    watcher.queue_removal(inbox)
    watcher.process_removals()

    assert rows("documents") == 1 and load_latest_chunks(config.DB_PATH)


def test_startup_removes_documents_deleted_while_watcher_was_off(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Keep.pdf", "Keep me."))
    ingest(make_pdf(config.DOCUMENTS_DIR / "Gone.pdf", "Remove me."))
    (config.ARCHIVE_DIR / "Gone.v1.pdf").unlink()

    removed = remove_missing_documents(config.DB_PATH)

    assert [r["key"] for r in removed] == ["Gone.pdf"]
    assert [d["filename"] for d in list_library(config.DB_PATH)] == [
        "Keep.pdf"
    ]


def test_startup_cleanup_skipped_without_archive_folder(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Revenue grew."))
    (config.ARCHIVE_DIR / "Report.v1.pdf").unlink()
    config.ARCHIVE_DIR.rmdir()

    assert remove_missing_documents(config.DB_PATH) == []
    assert rows("documents") == 1


def test_documents_processed_before_archiving_get_archived(project):
    # A CHUNKED document still in the inbox, as in older databases.
    path = make_pdf(config.DOCUMENTS_DIR / "Old.pdf", "Old document.")
    checksum = watcher.file_checksum(path)
    watcher.register_document(config.DB_PATH, path, checksum)
    record = watcher.get_latest_document(config.DB_PATH, "Old.pdf")
    watcher.process_document(path, record, approximate_token_count,
                             config.DB_PATH, config.PARSED_DIR)
    assert path.is_file()

    ingest(path)  # the startup scan

    assert (config.ARCHIVE_DIR / "Old.v1.pdf").is_file()
    assert not path.exists()


# --- chat sessions ------------------------------------------------------

def upload(tmp_path, name, text):
    folder = tmp_path / "upload_source"
    folder.mkdir(exist_ok=True)
    return make_pdf(folder / name, text)


def test_session_documents_are_private_and_deleted_at_end(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Library.pdf", "Library facts."))
    session = sessions.create_session(config.DB_PATH)

    result = sessions.add_session_document(
        session, upload(project, "Mine.pdf", "Private session facts."),
        config.DB_PATH, approximate_token_count,
    )
    assert result["status"] == "CHUNKED"

    library_text = " ".join(c["text"] for c in
                            load_latest_chunks(config.DB_PATH))
    session_text = " ".join(c["text"] for c in
                            load_latest_chunks(config.DB_PATH, session))
    assert "Library facts" in library_text
    assert "Private session" not in library_text
    assert "Private session" in session_text
    assert "Library facts" not in session_text

    load_or_compute_vectors(config.DB_PATH,
                            load_latest_chunks(config.DB_PATH, session),
                            FakeEncoder())
    ended = sessions.end_session(session, config.DB_PATH)

    assert len(ended["documents_deleted"]) == 1
    assert load_latest_chunks(config.DB_PATH, session) == []
    assert not (config.UPLOADS_DIR / session).exists()
    assert not sessions.session_exists(session, config.DB_PATH)
    assert rows("documents") == 1  # the library document survives
    assert rows("chunk_vectors") == 0
    assert "Library facts" in " ".join(
        c["text"] for c in load_latest_chunks(config.DB_PATH))


def test_session_upload_identical_to_library_document_is_reused(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Library.pdf", "Shared facts."))
    copy = upload(project, "Copy.pdf", "")
    copy.write_bytes((config.ARCHIVE_DIR / "Library.v1.pdf").read_bytes())
    session = sessions.create_session(config.DB_PATH)

    result = sessions.add_session_document(session, copy, config.DB_PATH,
                                           approximate_token_count)
    assert result["status"] == "REUSED"
    assert load_latest_chunks(config.DB_PATH, session)

    sessions.end_session(session, config.DB_PATH)
    assert load_latest_chunks(config.DB_PATH)  # library copy untouched
    assert (config.ARCHIVE_DIR / "Library.v1.pdf").is_file()


def test_session_filename_does_not_version_the_library(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Report.pdf", "Library report."))
    session = sessions.create_session(config.DB_PATH)
    sessions.add_session_document(
        session, upload(project, "Report.pdf", "Session report."),
        config.DB_PATH, approximate_token_count,
    )

    [library] = get_document_versions(config.DB_PATH, "Report.pdf")
    assert library["version"] == 1
    [own] = get_document_versions(config.DB_PATH,
                                  registry_key("Report.pdf", session))
    assert own["version"] == 1 and own["session_id"] == session


def test_same_upload_twice_is_processed_once(project):
    session = sessions.create_session(config.DB_PATH)
    source = upload(project, "Mine.pdf", "Session facts.")
    first = sessions.add_session_document(session, source, config.DB_PATH,
                                          approximate_token_count)
    second = sessions.add_session_document(session, source, config.DB_PATH,
                                           approximate_token_count)
    assert first["document_id"] == second["document_id"]
    assert second["status"] == "CHUNKED"
    assert rows("documents") == 1


def test_idle_sessions_are_cleaned_up(project):
    idle = sessions.create_session(config.DB_PATH)
    active = sessions.create_session(config.DB_PATH)
    sessions.add_session_document(
        idle, upload(project, "Old.pdf", "Old chat."),
        config.DB_PATH, approximate_token_count,
    )
    long_ago = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    with sqlite3.connect(config.DB_PATH) as connection:
        connection.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            (long_ago, idle),
        )

    ended = sessions.cleanup_expired_sessions(config.DB_PATH, idle_hours=24)

    assert [e["session_id"] for e in ended] == [idle]
    assert sessions.session_exists(active, config.DB_PATH)
    assert rows("documents") == 0


def test_upload_rejects_unknown_session_and_non_pdf(project):
    with pytest.raises(ValueError, match="No such session"):
        sessions.add_session_document("missing", upload(project, "a.pdf", "x"),
                                      config.DB_PATH)
    session = sessions.create_session(config.DB_PATH)
    text_file = project / "notes.txt"
    text_file.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        sessions.add_session_document(session, text_file, config.DB_PATH)


# --- adding from the UI (no watcher) ------------------------------------

def test_add_to_library_processes_and_archives(project):
    from docintel.ingestion.library import add_to_library
    source = upload(project, "Report.pdf", "Revenue grew 5 percent.")

    result = add_to_library(source, config.DB_PATH, approximate_token_count)

    assert result["status"] == "CHUNKED"
    assert (config.ARCHIVE_DIR / "Report.v1.pdf").is_file()
    assert not (config.DOCUMENTS_DIR / "Report.pdf").exists()  # watcher-safe
    assert list((config.UPLOADS_DIR / "library").iterdir()) == []
    assert source.is_file()  # the caller's file is left alone

    again = add_to_library(source, config.DB_PATH, approximate_token_count)
    assert again["message"] == "Already in the library."
    assert rows("documents") == 1


def test_add_to_library_rejects_duplicates_and_non_pdfs(project):
    from docintel.ingestion.library import add_to_library
    source = upload(project, "A.pdf", "Same contents.")
    add_to_library(source, config.DB_PATH, approximate_token_count)
    copy = project / "B.pdf"
    copy.write_bytes(source.read_bytes())

    assert add_to_library(copy, config.DB_PATH,
                          approximate_token_count)["status"] == "DUPLICATE"
    with pytest.raises(ValueError, match="Unsupported"):
        add_to_library(project / "notes.txt", config.DB_PATH)


def test_unreadable_upload_is_left_in_the_inbox(project):
    from docintel.ingestion.library import add_to_library
    blank = project / "Blank.pdf"
    document = fitz.open()
    document.new_page()  # nothing on it: nothing to index
    document.save(blank)

    result = add_to_library(blank, config.DB_PATH, approximate_token_count)

    assert result["status"] == "NEEDS_REVIEW"
    assert result["message"].startswith(
        "Not searchable: page 1: the document has no text.")
    assert (config.DOCUMENTS_DIR / "Blank.pdf").is_file()
    assert not any(config.ARCHIVE_DIR.iterdir())


def test_scanned_upload_without_ocr_explains_why(project, monkeypatch):
    from docintel.ingestion.library import add_to_library
    monkeypatch.setattr(config, "OCR_ENGINE", "none")
    scan = project / "Scan.pdf"
    document = fitz.open()
    page = document.new_page()
    pixmap = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 595, 842), False)
    pixmap.clear_with(220)
    page.insert_image(page.rect, stream=pixmap.tobytes("png"))
    document.save(scan)

    result = add_to_library(scan, config.DB_PATH, approximate_token_count)

    assert result["status"] == "NEEDS_REVIEW"
    assert "it is scanned and OCR is turned off" in result["message"]


def test_library_list_shows_the_document_type(project):
    ingest(make_pdf(config.DOCUMENTS_DIR / "Typed.pdf", "Typed facts."))
    assert list_library(config.DB_PATH)[0]["document_type"] == "digital"
