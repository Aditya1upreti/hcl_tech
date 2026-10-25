#!/usr/bin/env python
"""app.py - Streamlit UI for RAG-Based Document Q&A System

Features:
- st.cache_resource for embedding model and vector store
- st.session_state for chat history
- Sidebar: provider selection, model name, key status, file upload, restart button
- Main area: chat message history, query input, answer with source chunks
- Warning banner: "RAG systems may hallucinate. Verify with the source. Not for legal or medical advice."
- Retrieval-only banner: "Gemini API key missing. Add GEMINI_API_KEY to .env to enable generated answers."
- Source chunks displayed beside EVERY answer: file, page, similarity score
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Load environment variables and read config
from dotenv import load_dotenv
load_dotenv()

# Read LLM configuration from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").lower()

import streamlit as st
from pathlib import Path

# Import project modules - using updated packages and shared vectorstore helper
from src.ingest import get_chunk_id, ingest_files
from src.llm import get_provider_status, generate_answer
from src.qa import retrieve_chunks, answer_from_context, CHUNK_K

# Import shared vectorstore helper
from src.vectorstore import get_collection, init_collection

# Page config
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CACHE RESOURCES
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def load_resources():
    """Load embedding model and ChromaDB vector store. Cached for performance."""
    # Initialize the shared collection (only once)
    collection = init_collection()

    # We have the collection; the embeddings are handled by the
    # SentenceTransformerEmbeddingFunction embedded in the collection.
    # Return the collection for use in queries.
    return collection, None  # db is the collection, embeddings are internal


# ---------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": ..., "answer": ..., "source_info": ...}

    if "provider_status" not in st.session_state:
        st.session_state.provider_status = get_provider_status()

    if "vector_store_loaded" not in st.session_state:
        st.session_state.vector_store_loaded = False

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with provider selection and key status."""
    st.sidebar.title("🔧 Settings")

    # Provider selection - compute default index safely
    current_provider = st.session_state.provider_status.get("provider", "gemini")
    provider_options = ["gemini", "ollama", "retrieval-only"]
    if current_provider not in provider_options:
        current_provider = "gemini"
    default_index = provider_options.index(current_provider)

    llm_provider = st.sidebar.radio(
        "LLM Provider",
        options=provider_options,
        index=default_index,
        key="provider_radio",
        help="Select the LLM provider to use for generated answers",
    )

    # Update provider when changed
    if llm_provider != st.session_state.provider_status.get("provider"):
        # get_provider_status is already a pure function, just read the status
        st.session_state.provider_status = get_provider_status(
            llm_provider=llm_provider,
            gemini_api_key=GEMINI_API_KEY,
            gemini_model=GEMINI_MODEL,
        )
        st.rerun()

    # Display model and key status
    provider = st.session_state.provider_status.get("provider", "gemini")
    model = st.session_state.provider_status.get("model", GEMINI_MODEL)
    key_status = st.session_state.provider_status.get("key_status", "red")

    st.sidebar.markdown("**Model:**")
    st.sidebar.text(model)

    # Key status indicator
    if key_status == "green":
        st.sidebar.success("✓ API key present")
    elif key_status == "yellow":
        st.sidebar.warning("⚠ Using Ollama fallback")
    else:
        st.sidebar.error("✗ No API key - retrieval-only mode")

    st.sidebar.markdown("**Provider status:**")
    st.sidebar.caption(f"{provider} | {model}")

    # File upload section
    st.sidebar.markdown("--")
    st.sidebar.header("📎 Add Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload policy documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload additional documents to expand the knowledge base",
    )

    if st.sidebar.button("📥 Ingest Uploaded Documents", type="primary", use_container_width=True):
        _ingest_uploaded_files(uploaded_files)

    # Restart in retrieval-only mode button
    st.sidebar.markdown("--")
    if st.sidebar.button("🔄 Run in Retrieval-Only Mode", use_container_width=True):
        # Reset provider to retrieval-only
        st.session_state.provider_status = get_provider_status(
            llm_provider="retrieval-only",
        )
        st.sidebar.success("Running in retrieval-only mode")
        st.rerun()

    # Information banner
    st.sidebar.markdown("--")
    st.sidebar.info(
        "ℹ️ **About:** This is a RAG-based Q&A system for college policy documents. "
        "Embeddings are local (sentence-transformers). Answers require an LLM key or "
        "operating Ollama. System runs in retrieval-only mode without an API key."
    )

    return llm_provider


def _ingest_uploaded_files(uploaded_files):
    """Handle uploading and ingesting new documents."""
    from pathlib import Path

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for uploaded_file in uploaded_files:
        # Save temp file
        temp_path = data_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(temp_path)
        st.sidebar.success(f"Saved: {uploaded_file.name}")

    # Run ingestion using the new ingest_files API
    # This no longer temporarily overrides module-level state
    with st.spinner("Ingesting documents... This may take a moment."):
        try:
            from src.ingest import ingest_files as ingest_new

            # Save uploaded files to data/raw, then ingest
            ingested_paths = []
            for uploaded_file in uploaded_files:
                temp_path = data_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                ingested_paths.append(temp_path)

            # Ingest the files using the new API
            chunks_created = ingest_new(
                file_paths=ingested_paths,
                persist_dir="chroma_db",
                collection_name="policy_docs",
            )

            st.sidebar.success(f"Documents ingested successfully! {chunks_created} chunks created.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------------------

def main():
    """Main application loop."""
    # Initialize session state BEFORE any widget that reads it
    init_session_state()

    # Render sidebar and get provider
    llm_provider = render_sidebar()

    # Header
    st.title("📚 RAG Document Q&A System")
    st.caption("AI-based Q&A over college policy/guideline documents")

    # Always-visible warning banner
    st.warning(
        "⚠️ **RAG systems may hallucinate. Verify with the source. Not for legal or medical advice.**",
        icon="⚠️",
    )

    # Retrieval-only banner
    provider = st.session_state.provider_status.get("provider", "gemini")
    key_status = st.session_state.provider_status.get("key_status", "red")
    if provider == "retrieval-only" and key_status == "red":
        st.info(
            "🔍 **Gemini API key missing. Add GEMINI_API_KEY to .env to enable generated answers. "
            "System running in retrieval-only mode.**",
            icon="ℹ️",
        )

    # Load resources
    embeddings, db = load_resources()

    # Mark vector store as loaded
    st.session_state.vector_store_loaded = True

    # Check if there are documents in the vector store
    try:
        collection_metadata = db._collection.metadata
        total_docs = collection_metadata.get("count", 0)
    except Exception:
        total_docs = 0

    # Main chat area
    st.markdown("### 💬 Chat")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, chat_msg in enumerate(st.session_state.chat_history):
            if chat_msg["role"] == "user":
                st.chat_message("user").write(chat_msg["content"])
            else:
                # Display assistant message with answer and source chunks
                with st.chat_message("assistant"):
                    st.write(chat_msg["content"])
                    if chat_msg.get("source_info"):
                        with st.expander(f"📄 Source chunks ({len(chat_msg['source_chunks'])} found)"):
                            for si in chat_msg["source_info"]:
                                st.write(si)

    # Query input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.chat_input("Ask a question about the policy documents...")
    with col2:
        clear_btn = st.button("Clear", help="Clear chat history")

    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()

    if query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        # Retrieve chunks
        with st.spinner("🔍 Retrieving relevant chunks..."):
            retrieved = retrieve_chunks(query, k=CHUNK_K)

        # Generate answer
        with st.spinner("🤔 Generating answer..."):
            # Check if we have an LLM available
            if provider == "retrieval-only" or key_status == "red":
                # Retrieval-only mode
                from src.llm import get_active_provider
                get_active_provider()  # Ensure provider is set correctly
                answer_result = answer_from_context(query, k=CHUNK_K)
                answer = answer_result["answer"]
                source_info = answer_result.get("source_info", "No source chunks found.")
                # Format source chunks for the expander
                source_chunks_display = []
                if retrieved:
                    for chunk in retrieved:
                        source_chunks_display.append(
                            f"• {chunk['source']} (page {chunk['page']}) — similarity: {chunk['similarity_pct']}"
                        )
                else:
                    source_chunks_display = ["No relevant chunks found."]

                response_time = answer_result.get("response_time", 0)

            else:
                # Try LLM-generated answer
                try:
                    answer_result = generate_answer(
                        query,
                        [c["content"] for c in retrieved],
                        llm_provider=provider,
                        gemini_api_key=GEMINI_API_KEY,
                        gemini_model=GEMINI_MODEL,
                    )
                    answer = answer_result.get(
                        "answer", "I don't know based on the provided documents."
                    )
                    response_time = answer_result.get("response_time", 0)
                    source_info = answer_result.get("source_info", "")
                except Exception as e:
                    # LLM call failed - fall back to retrieval-only
                    print(f"LLM error: {e}")
                    # Force retrieval-only mode
                    from src.llm import get_active_provider
                    get_active_provider()
                    answer_result = answer_from_context(query, k=CHUNK_K)
                    answer = answer_result["answer"]
                    source_info = answer_result.get("source_info", "No source chunks found.")
                    response_time = answer_result.get("response_time", 0)

        # Add assistant message to chat history
        source_chunks_for_expander = []
        if retrieved:
            for chunk in retrieved:
                source_chunks_for_expander.append(
                    f"📄 {chunk['source']} (page {chunk['page']}) — similarity: {chunk['similarity_pct']}"
                )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "source_info": source_chunks_for_expander,
            "response_time": response_time,
        })

        # Rerun to display the new message
        st.rerun()


if __name__ == "__main__":
    main()