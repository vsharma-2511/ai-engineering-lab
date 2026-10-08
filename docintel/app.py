"""DocIntel chat: ask questions about your PDFs, with checked citations.

    python -m streamlit run app.py

Chat page:    ask about the library, or about files uploaded into this
              chat only (deleted when you end the chat).
Library page: see, add and remove the documents you keep.
"""
import html
from pathlib import Path
import tempfile
import time

import pandas as pd
import streamlit as st

from docintel import config, sessions
from docintel.ingestion.library import add_to_library, delete_document, list_library
from docintel.qa.chat import chat_turn
from docintel.qa.llm import PROVIDERS, LLMError, get_llm

st.set_page_config(page_title="DocIntel", page_icon="📄", layout="wide")

st.markdown("""
<style>
.citation {border-left: 3px solid var(--primary-color, #4c6ef5);
           padding: .35rem .75rem; margin: .35rem 0 .6rem;
           background: rgba(127,127,127,.08); border-radius: 0 6px 6px 0;
           font-size: .92rem; white-space: pre-wrap;}
.citation-head {font-size: .8rem; opacity: .75; margin-bottom: .15rem;}
</style>
""", unsafe_allow_html=True)

STATUS_ICONS = {"CHUNKED": "✅", "REUSED": "♻️", "NEEDS_REVIEW": "⚠️",
                "FAILED": "❌", "DUPLICATE": "ℹ️"}


# --- one-time and cached resources ------------------------------------

@st.cache_resource(show_spinner=False)
def startup_cleanup() -> int:
    """End chats idle past SESSION_IDLE_HOURS (once per server start)."""
    return len(sessions.cleanup_expired_sessions())


@st.cache_resource(show_spinner=False)
def load_llm(provider: str, model: str):
    return get_llm(provider, model)


def init_state() -> None:
    defaults = {
        "messages": [],        # {"role", "content", "result"?, "error"?}
        "session_id": None,    # created on the first upload into this chat
        "chat_files": {},      # filename -> add_session_document() result
        "seen_uploads": set(), # UploadedFile ids already processed
        "uploader_key": 0,     # bumped to clear the file uploader
        "confirm_remove": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def safe(text: str) -> str:
    """Markdown-safe: Streamlit renders $...$ as math."""
    return text.replace("$", "\\$")


# --- rendering --------------------------------------------------------

def render_result(result: dict) -> None:
    status = result["status"]
    if status == "not_found":
        st.info("The documents don't contain an answer to this question.",
                icon="🔎")
    else:
        if status == "unsupported":
            st.warning("None of the quotes the model gave could be found in "
                       "the sources, so don't rely on this answer.",
                       icon="⚠️")
        st.markdown(safe(result["answer"]))

    for citation in result["citations"]:
        quote = html.escape(citation["quote"]).replace("$", "&#36;")
        source = html.escape(citation["document"])
        st.markdown(
            f'<div class="citation"><div class="citation-head">'
            f'[{citation["source"]}] {source} · page {citation["page"]}'
            f'</div>“{quote}”</div>',
            unsafe_allow_html=True,
        )

    details = []
    if result.get("searched_for") and result["searched_for"] != result.get("asked"):
        details.append(f"searched for: *{safe(result['searched_for'])}*")
    details.append(f"{result['provider']} / {result['model']}")
    if result.get("seconds") is not None:
        details.append(f"{result['seconds']:.0f}s")
    st.caption(" · ".join(details))

    if result["sources"]:
        with st.expander(f"Sources searched ({len(result['sources'])})"):
            for source in result["sources"]:
                st.markdown(f"[{source['source']}] {safe(source['document'])}"
                            f" · page {source['page']}")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["error"])
        elif message.get("result"):
            render_result(message["result"])
        else:
            st.markdown(safe(message["content"]))


def history_for_rewrite() -> list[dict]:
    """Earlier turns as plain text, skipping errors."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if not m.get("error") and m["content"]
    ]


# --- shared sidebar: model settings -----------------------------------

def model_settings() -> tuple[object | None, dict]:
    """Model choice and answer settings, shown on every page."""
    with st.sidebar.expander("⚙️ Model and settings"):
        providers = sorted(PROVIDERS)
        provider = st.selectbox(
            "Provider", providers,
            index=providers.index(config.QA_PROVIDER)
            if config.QA_PROVIDER in providers else 0,
            help="ollama runs on this computer: free and private.",
        )
        default_model = (config.QA_MODEL if provider == config.QA_PROVIDER
                         and config.QA_MODEL else
                         config.QA_DEFAULT_MODELS[provider])
        model = st.text_input("Model", value=default_model)
        settings = {
            "rewrite": st.toggle(
                "Understand follow-up questions", value=True,
                help="Rewrites e.g. 'and at Central?' into a full question "
                     "using the conversation. One extra model call per "
                     "follow-up.",
            ),
            "top_k": st.slider("Sources per question", 3, 8, config.QA_TOP_K),
        }

    llm = None
    try:
        llm = load_llm(provider, model.strip())
        st.sidebar.caption(f"Model: {provider} / {model.strip()}")
    except LLMError as exc:
        st.sidebar.error(str(exc))
    return llm, settings


# --- chat page --------------------------------------------------------

def chat_files_sidebar() -> str:
    """Uploads for this chat only, search scope and End chat."""
    st.sidebar.divider()
    st.sidebar.subheader("This chat's files")
    st.sidebar.caption("Private to this chat and deleted when you end it.")
    uploads = st.sidebar.file_uploader(
        "Add PDFs", type=["pdf"], accept_multiple_files=True,
        key=f"chat_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    for upload in uploads or []:
        if upload.file_id in st.session_state.seen_uploads:
            continue
        st.session_state.seen_uploads.add(upload.file_id)
        if st.session_state.session_id is None:
            st.session_state.session_id = sessions.create_session()
        with st.sidebar.status(f"Processing {upload.name}… (scanned pages take ~20 s each)") as status:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / upload.name
                path.write_bytes(upload.getvalue())
                try:
                    result = sessions.add_session_document(
                        st.session_state.session_id, path)
                except Exception as exc:  # show it, keep the app running
                    result = {"status": "FAILED", "filename": upload.name,
                              "message": str(exc)}
            st.session_state.chat_files[result["filename"]] = result
            ok = result["status"] in {"CHUNKED", "REUSED"}
            status.update(label=f"{upload.name}: {result['message']}",
                          state="complete" if ok else "error")
            if ok:  # searching a new upload is almost always the intent
                st.session_state.scope = "This chat's files"

    for name, result in st.session_state.chat_files.items():
        icon = STATUS_ICONS.get(result["status"], "•")
        st.sidebar.markdown(f"{icon} {safe(name)}")
        if result["status"] not in {"CHUNKED", "REUSED"}:
            st.sidebar.caption(result["message"])

    searchable = any(r["status"] in {"CHUNKED", "REUSED"}
                     for r in st.session_state.chat_files.values())
    scopes = ["This chat's files", "Library"] if searchable else ["Library"]
    if st.session_state.get("scope") not in scopes:
        st.session_state.scope = scopes[0]
    scope = st.sidebar.radio("Search in", scopes, horizontal=True,
                             key="scope")

    st.sidebar.divider()
    with st.sidebar.popover("End chat", width="stretch"):
        st.markdown("Clears the conversation"
                    + (" and **deletes this chat's files** with everything "
                       "derived from them." if st.session_state.chat_files
                       else "."))
        if st.button("End chat", type="primary", width="stretch"):
            if st.session_state.session_id:
                sessions.end_session(st.session_state.session_id)
            for key in ("messages", "session_id", "chat_files",
                        "seen_uploads", "scope"):
                st.session_state.pop(key, None)
            st.session_state.uploader_key += 1
            st.rerun()
    return scope


def chat_page(llm, settings: dict) -> None:
    in_chat = chat_files_sidebar() == "This chat's files"
    library_size = sum(d["status"] == "CHUNKED"
                       for d in list_library(config.DB_PATH))
    if in_chat:
        count = sum(r["status"] in {"CHUNKED", "REUSED"}
                    for r in st.session_state.chat_files.values())
        st.caption(f"Searching **this chat's files** ({count})")
    else:
        st.caption(f"Searching **the library** ({library_size} documents)")

    if not st.session_state.messages:
        st.markdown("### Ask a question about your documents")
        st.markdown(
            "Answers come only from your documents, and every claim is "
            "backed by a quote that is checked against the page it cites.\n\n"
            "- Upload PDFs for this chat in the sidebar, or use the "
            "**Library** page.\n"
            "- Follow-ups like *“and in 2024?”* work too."
        )
        if llm is None:
            st.info("Choose a working model in the sidebar to start.")
        elif not in_chat and library_size == 0:
            st.info("The library is empty: add a PDF on the Library page or "
                    "upload one to this chat in the sidebar.")

    for message in st.session_state.messages:
        render_message(message)

    # At the top level of the page, so it stays pinned to the bottom.
    question = st.chat_input(
        "Ask about your documents…",
        disabled=llm is None or (not in_chat and library_size == 0),
    )
    if not question:
        return

    history = history_for_rewrite()
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        with st.spinner("Reading the sources… local models can take "
                        "20–60 seconds."):
            started = time.monotonic()
            try:
                result = chat_turn(
                    question, history, llm,
                    session_id=st.session_state.session_id if in_chat else None,
                    top_k=settings["top_k"], rewrite=settings["rewrite"],
                )
                result["seconds"] = time.monotonic() - started
                message = {"role": "assistant", "result": result,
                           "content": result["answer"]
                           or "The documents don't contain an answer."}
            except Exception as exc:  # show it, keep the conversation
                message = {"role": "assistant", "content": "",
                           "error": f"Could not answer: {exc}"}
    st.session_state.messages.append(message)
    st.rerun()


# --- library page -----------------------------------------------------

def library_page() -> None:
    st.markdown("### Library")
    st.caption("Documents you keep. Processed originals live in `archive/`; "
               "you can also drop PDFs into `documents/` while the watcher "
               "runs.")

    with st.form("add_to_library", clear_on_submit=True, border=True):
        files = st.file_uploader("Add PDFs to the library", type=["pdf"],
                                 accept_multiple_files=True)
        if st.form_submit_button("Add to library", type="primary") and files:
            for upload in files:
                with st.status(f"Processing {upload.name}… (scanned pages take ~20 s each)") as status:
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder) / upload.name
                        path.write_bytes(upload.getvalue())
                        try:
                            result = add_to_library(path)
                        except Exception as exc:
                            result = {"status": "FAILED", "message": str(exc)}
                    status.update(
                        label=f"{upload.name}: {result['message']}",
                        state="complete" if result["status"] == "CHUNKED"
                        else "error",
                    )

    documents = list_library(config.DB_PATH)
    if not documents:
        st.info("The library is empty.")
        return

    st.dataframe(
        pd.DataFrame([{
            "Document": d["filename"], "Version": d["version"],
            "Type": d.get("document_type") or "",
            "Status": f"{STATUS_ICONS.get(d['status'], '')} {d['status']}",
            "Chunks": d["chunks"], "File": d["location"],
        } for d in documents]),
        hide_index=True, width="stretch",
    )

    with st.expander("Remove a document"):
        name = st.selectbox("Document", [d["filename"] for d in documents])
        st.caption("Deletes every version with its chunks, vectors, parsed "
                   "output and PDF. This cannot be undone.")
        if st.session_state.confirm_remove != name:
            if st.button("Remove…"):
                st.session_state.confirm_remove = name
                st.rerun()
        else:
            st.warning(f"Remove **{safe(name)}** permanently?")
            left, right = st.columns(2)
            if left.button("Yes, remove it", type="primary", width="stretch"):
                summary = delete_document(config.DB_PATH, name)
                st.session_state.confirm_remove = None
                st.toast(f"Removed {name} "
                         f"({summary['chunks'] if summary else 0} chunks).")
                st.rerun()
            if right.button("Cancel", width="stretch"):
                st.session_state.confirm_remove = None
                st.rerun()


def main() -> None:
    init_state()
    ended = startup_cleanup()
    if ended:
        st.toast(f"Cleaned up {ended} idle chat(s).")

    page = st.navigation([
        st.Page(lambda: chat_page(llm, settings), title="Chat", icon="💬",
                url_path="chat", default=True),
        st.Page(library_page, title="Library", icon="📚", url_path="library"),
    ])
    st.sidebar.title("📄 DocIntel")
    llm, settings = model_settings()
    page.run()


main()
