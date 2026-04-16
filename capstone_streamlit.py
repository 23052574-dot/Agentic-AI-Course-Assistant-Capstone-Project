"""
capstone_streamlit.py — Agentic AI Course Assistant
Domain  : Agentic AI Hands-On Course (13-day curriculum)
User    : B.Tech 4th-year students
"""

import uuid

import streamlit as st

from agent import DOCUMENTS, build_agent

st.set_page_config(
    page_title="Agentic AI Course Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic AI Course Assistant")
st.caption(
    "Ask anything about the 13-day Agentic AI Hands-On Course — "
    "concepts, code patterns, session content, and submission requirements."
)

@st.cache_resource
def load_agent():
    """
    Build LLM, ChromaDB KB, and compiled LangGraph app once.
    @st.cache_resource ensures this function is NOT re-executed on every
    Streamlit rerun (which happens on every user interaction).
    """
    app, embedder, collection = build_agent()
    return app, embedder, collection


try:
    agent_app, _, collection = load_agent()
    st.success(f"✅ Course knowledge base loaded — {collection.count()} session documents")
except Exception as exc:
    st.error(f"❌ Failed to load agent: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant answers questions about the "
        "**Agentic AI Hands-On Course**"
        "It retrieves answers from 13 session documents covering "
        "every day of the curriculum."
    )
    st.write(f"**Session ID:** `{st.session_state.thread_id}`")
    st.divider()

    st.write("**Topics covered:**")
    for doc in DOCUMENTS:
        st.write(f"• {doc['topic']}")

    st.divider()
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about a session, concept, or code pattern…"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer") or "Sorry, I could not generate a response."

        st.write(answer)

        faith   = result.get("faithfulness") or 0.0
        route   = result.get("route") or "—"
        sources = result.get("sources") or []
        if faith > 0:
            st.caption(
                f"Faithfulness: {faith:.2f} | Route: {route} | "
                f"Sources: {', '.join(sources) if sources else '—'}"
            )

    st.session_state.messages.append({"role": "assistant", "content": answer})
