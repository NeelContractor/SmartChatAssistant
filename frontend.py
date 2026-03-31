import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    checkpointer,
    ingest_pdf,
    retrieve_all_threads,
    thread_docuemnt_metadata,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartChat Assistant",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* App background */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* App title in sidebar */
.app-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    padding: 0.2rem 0 0.5rem 0;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 1rem;
}
.app-title span { color: #6ee7b7; }

/* Thread ID badge */
.thread-badge {
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #94a3b8 !important;
    word-break: break-all;
    margin-bottom: 0.8rem;
    display: block;
}

/* Conversation card — full width, stacks cleanly */
.conv-card {
    background: #1a2235;
    border: 1px solid #1f2d44;
    border-radius: 10px;
    padding: 8px 12px;
    margin-bottom: 4px;
    width: 100%;
    box-sizing: border-box;
}
.conv-card.active {
    border-color: #6ee7b7;
    background: #1a2e28;
}
.conv-card-id {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #475569 !important;
    margin-bottom: 2px;
}
.conv-card-preview {
    font-size: 0.76rem;
    color: #94a3b8 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}

/* Section label */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #475569 !important;
    margin: 0.8rem 0 0.4rem 0;
}

/* Main page title */
.main-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 2rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    margin-bottom: 0.1rem;
}
.main-title span { color: #6ee7b7; }

.main-subtitle {
    font-size: 0.82rem;
    color: #64748b;
    margin-bottom: 1.2rem;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #161b27 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    margin-bottom: 6px !important;
}

/* Primary buttons */
.stButton > button {
    background: #1e2535 !important;
    color: #cbd5e1 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: #6ee7b7 !important;
    color: #0f1117 !important;
    border-color: #6ee7b7 !important;
}

/* New chat button accent */
.new-chat-btn > div > button {
    background: #6ee7b7 !important;
    color: #0f1117 !important;
    border-color: #6ee7b7 !important;
    font-weight: 600 !important;
}
.new-chat-btn > div > button:hover {
    background: #34d399 !important;
    border-color: #34d399 !important;
}

/* Conv action row: Open + Delete side by side, full width */
.conv-actions {
    display: flex;
    gap: 6px;
    margin-top: 4px;
    width: 100%;
}

/* Open button inside conv-actions */
.conv-open-btn > div > button {
    background: transparent !important;
    color: #6ee7b7 !important;
    border: 1px solid #2a3d35 !important;
    border-radius: 6px !important;
    font-size: 0.72rem !important;
    padding: 2px 10px !important;
    height: 28px !important;
    min-height: 0 !important;
    width: 100% !important;
}
.conv-open-btn > div > button:hover {
    background: #6ee7b7 !important;
    color: #0f1117 !important;
    border-color: #6ee7b7 !important;
}

/* Delete button inside conv-actions */
.conv-del-btn > div > button {
    background: transparent !important;
    color: #f87171 !important;
    border: 1px solid #3d1f1f !important;
    border-radius: 6px !important;
    font-size: 0.72rem !important;
    padding: 2px 8px !important;
    height: 28px !important;
    min-height: 0 !important;
    width: 100% !important;
}
.conv-del-btn > div > button:hover {
    background: #f87171 !important;
    color: #0f1117 !important;
    border-color: #f87171 !important;
}

/* Active thread — only show delete, no Open */
.conv-del-only > div > button {
    background: transparent !important;
    color: #f87171 !important;
    border: 1px solid #3d1f1f !important;
    border-radius: 6px !important;
    font-size: 0.72rem !important;
    padding: 2px 8px !important;
    height: 28px !important;
    min-height: 0 !important;
}
.conv-del-only > div > button:hover {
    background: #f87171 !important;
    color: #0f1117 !important;
    border-color: #f87171 !important;
}

/* Clear button */
.clear-btn .stButton > button {
    background: transparent !important;
    color: #64748b !important;
    border: 1px solid #2d3748 !important;
    font-size: 0.75rem !important;
}
.clear-btn .stButton > button:hover {
    color: #f87171 !important;
    border-color: #f87171 !important;
    background: transparent !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a2235 !important;
    border-radius: 10px !important;
}

/* Divider */
hr { border-color: #1f2937 !important; }

/* Caption */
.stCaption { color: #64748b !important; font-size: 0.72rem !important; }

/* Confirmation warning */
.stAlert { border-radius: 8px !important; font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["confirm_delete"] = None


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])


def get_conversation_preview(thread_id):
    """Return the first human message as a short preview string."""
    try:
        messages = load_conversation(thread_id)
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                preview = str(msg.content).strip().replace("\n", " ")
                return preview[:55] + ("…" if len(preview) > 55 else "")
    except Exception:
        pass
    return "Empty conversation"


def delete_thread(thread_id):
    """
    Remove thread from sidebar and purge its checkpoints from the DB.

    Uses checkpointer.delete_thread() (the official LangGraph API) which
    deletes from both the `checkpoints` and `writes` tables correctly.
    Falls back to direct SQL only if the method is unavailable (older versions).
    """
    tid = str(thread_id)

    # 1. Remove from in-memory session state
    st.session_state["chat_threads"] = [
        t for t in st.session_state["chat_threads"] if str(t) != tid
    ]
    st.session_state["ingested_docs"].pop(tid, None)

    # 2. Purge from the SQLite checkpoint store
    try:
        # Preferred: use the official API (LangGraph ≥ 0.2)
        config = {"configurable": {"thread_id": tid}}
        checkpointer.delete_thread(config)
    except (AttributeError, TypeError):
        # Fallback for older LangGraph versions — correct table names are
        # `checkpoints` and `writes` (NOT `checkpoint_writes`)
        try:
            from backend import conn as db_conn
            cur = db_conn.cursor()
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
            cur.execute("DELETE FROM writes WHERE thread_id = ?", (tid,))
            db_conn.commit()
        except Exception:
            pass

    # 3. If the deleted thread was active, start a fresh chat
    if str(st.session_state["thread_id"]) == tid:
        reset_chat()


# ---------------------------------------------------------------------------
# Session Initialization
# ---------------------------------------------------------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "confirm_delete" not in st.session_state:
    st.session_state["confirm_delete"] = None

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="app-title">SmartChat <span>Assistant</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="thread-badge">Thread · {thread_key[:20]}…</span>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True):
        reset_chat()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # --- PDF section ---
    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.success(
            f"📄 **{latest_doc.get('filename')}**  \n"
            f"{latest_doc.get('chunks')} chunks · {latest_doc.get('documents')} pages"
        )
    else:
        st.info("No PDF indexed for this chat.")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF", type=["pdf"], label_visibility="collapsed"
    )
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.info(f"`{uploaded_pdf.name}` already processed.")
        else:
            with st.status("Indexing PDF…", expanded=True) as status_box:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(
                    label="✅ PDF indexed", state="complete", expanded=False
                )

    st.divider()

    # --- Conversations list ---
    st.markdown(
        '<div class="section-label">Conversations</div>', unsafe_allow_html=True
    )

    if not threads:
        st.caption("No past conversations yet.")
    else:
        for thread_id in threads:
            tid_str = str(thread_id)
            is_active = tid_str == thread_key
            preview = get_conversation_preview(thread_id)
            short_id = tid_str[:8] + "…"
            card_cls = "conv-card active" if is_active else "conv-card"

            # Card (preview text) — full width, no columns
            st.markdown(
                f"""<div class="{card_cls}">
                    <div class="conv-card-id">{short_id}</div>
                    <div class="conv-card-preview">{preview}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Action buttons below the card
            if is_active:
                # Active thread — only a delete button, full width
                st.markdown('<div class="conv-del-only">', unsafe_allow_html=True)
                if st.button("✕ Delete", key=f"del-{tid_str}", use_container_width=True):
                    st.session_state["confirm_delete"] = tid_str
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Inactive thread — Open (wider) + Delete (narrower) side by side
                col_open, col_del = st.columns([3, 1])
                with col_open:
                    st.markdown('<div class="conv-open-btn">', unsafe_allow_html=True)
                    if st.button("Open", key=f"load-{tid_str}", use_container_width=True):
                        selected_thread = thread_id
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_del:
                    st.markdown('<div class="conv-del-btn">', unsafe_allow_html=True)
                    if st.button("✕", key=f"del-{tid_str}", use_container_width=True):
                        st.session_state["confirm_delete"] = tid_str
                    st.markdown("</div>", unsafe_allow_html=True)

            # Add a small spacer between entries
            st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

        # Confirmation prompt
        if st.session_state["confirm_delete"]:
            pending = st.session_state["confirm_delete"]
            st.warning(f"Delete `{pending[:8]}…`? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete", key="confirm-yes", use_container_width=True):
                    delete_thread(pending)
                    st.session_state["confirm_delete"] = None
                    st.rerun()
            with c2:
                if st.button("Cancel", key="confirm-no", use_container_width=True):
                    st.session_state["confirm_delete"] = None
                    st.rerun()


# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">SmartChat <span>Assistant</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subtitle">'
    "Ask anything · Search the web · Analyse your PDF · Run calculations"
    "</div>",
    unsafe_allow_html=True,
)

# Clear button (only shown when there are messages)
if st.session_state["message_history"]:
    _, col_clear = st.columns([8, 1])
    with col_clear:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑 Clear", help="Clear chat display (does not delete history)"):
            st.session_state["message_history"] = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# Render chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask about your document, search the web, or calculate…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_docuemnt_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 {doc_meta.get('filename')}  ·  "
            f"{doc_meta.get('chunks')} chunks  ·  {doc_meta.get('documents')} pages"
        )

st.divider()

# ---------------------------------------------------------------------------
# Load past conversation on sidebar click
# ---------------------------------------------------------------------------

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.session_state["confirm_delete"] = None
    st.rerun()