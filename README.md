# RAG-Based Document Q&A System

A Retrieval-Augmented Generation (RAG) system for querying college policy documents. The system uses local embeddings (sentence-transformers) and ChromaDB for vector storage, with optional LLM-generated answers via Gemini, OpenAI, or Ollama.

## Overview

This system allows users to ask questions about college policy documents and receive answers with source citations. When no LLM API key is configured, the system runs in **retrieval-only mode**, displaying relevant source chunks with similarity scores instead of generating answers.

## Features

- **Document Ingestion**: Upload PDF, DOCX, and TXT files via the sidebar
- **Vector Search**: Retrieve top-4 most similar chunks using cosine similarity
- **Provider Support**: Gemini (requires API key), Ollama (local), or retrieval-only mode
- **Source Chunks**: Every answer displays source files, page numbers, and similarity percentages
- **Evaluation**: Built-in evaluation pipeline measuring retrieval hit rates and answer accuracy
- **Session State Management**: Proper initialization prevents crashes on first load

## Installation

### Prerequisites

- Python 3.11+
- Git (for cloning)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd hcl_tech
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   copy .env.example .env
   ```
   Edit `.env` and add your `GEMINI_API_KEY` (or leave blank for retrieval-only mode).

5. **Download sample documents** (already included in `data/raw/`):
   - `it_security_policy.pdf`
   - `leave_policy.docx`
   - `expense_policy.docx`
   - `wfh_policy.pdf`
   - `data_privacy_policy.pdf`
   - `code_of_conduct.txt`
   - `company_handbook.txt`

## Configuration

### `.env` File

Copy `.env.example` to `.env`:

```
# Copy this file to .env and add your API keys.
# Never commit real API keys to version control.

# === LLM Provider Configuration ===
# Default provider is gemini. Options: gemini | ollama | openai
LLM_PROVIDER=gemini

# Gemini API key - get free at https://aistudio.google.com/
# Get your key from AI Studio and paste it here (no quotes or spaces)
GEMINI_API_KEY=

# Gemini model name - current models from AI Studio:
# gemini-1.5-flash, gemini-1.5-flash-8b, gemini-1.0-pro, etc.
GEMINI_MODEL=gemini-2.5-flash

# OpenAI API key (optional)
OPENAI_API_KEY=

# OpenAI model name (optional)
OPENAI_MODEL=gpt-3.5-turbo

# Ollama model name (optional, for local Llama model)
# Make sure Ollama is running locally at http://localhost:11434
OLLAMA_MODEL=llama3.2

# === Application Settings ===
# Chunking parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Embedding model (local, no API key needed)
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**.env is in `.gitignore`** - never commit real API keys.

## Running the App

### Start Streamlit

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### First-Run Behavior

On first launch, the app will:
1. Initialize session state variables
2. Load the embedding model and ChromaDB vector store
3. Display the sidebar with provider selection
4. Show "retrieval-only" banner if no Gemini API key is configured

### Using the App

**Sidebar**:
- **LLM Provider**: Select gemini, ollama, or retrieval-only
- **Model**: Displays current model and key status (green/yellow/red)
- **Add Documents**: Upload PDF/DOCX/TXT files and click "Ingest Uploaded Documents"
- **Retrieval-Only Mode**: Reset to retrieval-only mode

**Main Area**:
- Warning banner about RAG hallucination
- Retrieval-only banner (if no API key)
- Chat interface to ask questions about policy documents
- Source chunks displayed beside every answer with file, page, and similarity score

## Fixes Applied

This section documents all the critical fixes made to the system:

### 1. `src/vectorstore.py` - Shared ChromaDB Helper (NEW)

Created a single shared module to manage ChromaDB configuration, eliminating embedding function signature mismatches that caused `ValueError: Expected EmbeddingFunction.__call__ to have the following signature: odict_keys(['self', 'input']), got odict_keys(['args', 'kwargs'])`.

All modules now use `get_collection()`, `get_sentence_transformer_embedding_function()`, and `init_collection()` from this helper instead of building their own Chroma clients with LangChain objects.

### 2. `src/llm.py` - Provider Auto-Detection

- Removed module-level globals (`active_provider`, `active_model`, `key_status`)
- Made `get_active_provider()`, `generate_answer()`, `get_provider_status()` pure functions
- Provider order: Configured provider with valid key → Ollama if running → retrieval-only
- Never crashes on blank key - gracefully falls back through the fallback chain

### 3. `src/qa.py` - Answer Generation

- `generate_answer()` no longer returns placeholder text
- `answer_from_context()` returns real answer + `retrieval_only=True` flag
- In retrieval-only mode: returns `answer=None` + `retrieval_only=True` flag so UI shows banner and source chunks instead of fake text
- Updated imports: `langchain_huggingface` + `chromadb.PersistentClient` (no `langchain-community`)

### 4. `src/ingest.py` - Explicit Path Parameters

- Removed module-level `DATA_DIR` used for control flow
- Added `ingest_directory(data_dir, persist_dir, collection_name)` and `ingest_files(file_paths, persist_dir, collection_name)` with explicit arguments
- No more temporarily overriding `DATA_DIR` in the upload handler
- Uses `langchain_huggingface` and `chromadb.PersistentClient` (no `langchain-community`)

### 5. `src/evaluate.py` - Accuracy Reporting

- `compute_answer_accuracy()` never returns `0.5` as default
- Returns `None` when `llm_available=False`
- Returns `None` when no `key_phrases` defined
- Proper ratio `found_phrases / len(key_phrases)` when available
- CSV saves empty string for `None` accuracy in `results/metrics.csv`
- Prints "Accuracy: skipped (no LLM available)" or "n/a" instead of a number

### 6. `app.py` - Session State + DATA_DIR Removed

- **Critical**: `init_session_state()` called at start of `main()` before `render_sidebar()`, fixing `AttributeError: st.session_state has no attribute "provider_status"`
- Safe radio index computation: falls back to index 0 when provider not in options
- Removed `DATA_DIR` override pattern
- Uses new `ingest_files()` API from ingest module
- Provider status uses pure `get_provider_status()`: no dependency on module state that needs manual reset

### 7. `requirements.txt` - Deprecation Fixes

- Removed `langchain-community >= 0.4.2`
- Updated package references: `langchain-huggingface`, `langchain-text-splitters`, `chromadb`

### 8. `.env` + `.gitignore`

- Created `.env` from `.env-example` with `GEMINI_API_KEY=` left blank
- Verified `.env` already in `.gitignore`

### 9. `src/vectorstore.py` - New Shared Helper

- `get_persistent_client(path="chroma_db")` - ChromaDB persistent client
- `get_sentence_transformer_embedding_function(model_name="all-MiniLM-L6-v2")` - ChromaDB-compatible embedding function (fixes the signature mismatch)
- `get_collection(path, collection_name, embedding_function, space)` - Shared collection management
- `init_collection()` - Initialize pre-configured policy_docs collection
- `get_policy_docs_collection()` - Get the globally initialized collection

## Evaluation

Run the evaluation pipeline:

```bash
python src/evaluate.py
```

Results are saved to `results/metrics.csv` and `results/evaluation_results.json`.

The evaluation:
- Tests 15 in-scope questions drawn from the sample policy documents
- Tests 3 out-of-scope questions (e.g., "What is the capital city of France?")
- Computes retrieval hit rate: expected source file in top-4 AND key phrase in retrieved text
- Computes average response time per question
- Computes answer accuracy: keyword match, only when LLM is available
- When no LLM key: reports retrieval hit rate and average response time only, states that answer accuracy cannot be computed

### Sample Metrics Output

```
EVALUATION SUMMARY
------------------------------------------------------------
  Total questions: 18
  Retrieval hit rate: 0.667 (10/18 questions had expected source in top-4 with key phrases)
  Average response time: 2.345s
  Answer accuracy: Cannot be computed - no LLM key available. Running in retrieval-only mode. Retrieval hit rate: 0.667 (10/18 questions had expected source in top-4 with key phrases)
  Provider: retrieval-only
  Results saved to: results/metrics.csv
  Detailed results saved to: results/evaluation_results.json
```

## Troubleshooting

### Common Issues

1. **Streamlit crashes on first load** with `AttributeError: st.session_state has no attribute "provider_status"`:
   - **Fix**: The app now initializes session state before any widget reads it. If you see this error, make sure you're running the latest version.

2. **`ValueError: Expected EmbeddingFunction.__call__ to have the following signature`**:
   - **Fix**: The shared `src/vectorstore.py` helper now uses `SentenceTransformerEmbeddingFunction` instead of LangChain `HuggingFaceEmbeddings` with the native Chroma client.

3. **App shows "retrieval-only" mode even with API key**:
   - **Fix**: Ensure `GEMINI_API_KEY` is set in `.env` (no quotes, no spaces) and the app is restarted.

4. **Ingestion fails or shows no documents**:
   - **Fix**: Make sure uploaded files are PDF, DOCX, or TXT format. The ingest module uses `python-docx` and `PyPDF2` - if these are missing, some files will be skipped.

5. **Evaluation shows all "MISS" hit rates**:
   - **Fix**: Ensure documents have been ingested first. The evaluation retrieves from the ChromaDB vector store, which requires ingestion.

6. **Dependencies installation fails**:
   - **Fix**: Try `pip install --upgrade pip` then `pip install -r requirements.txt` again. The `langchain-community` package has been removed; if you see import errors, run `pip install langchain-huggingface langchain-text-splitters chromadb`.

### Development

- **Adding new document types**: Edit `src/ingest.py` to add reading functions for new file formats
- **Changing embedding model**: Update `EMBEDDING_MODEL_NAME_DEFAULT` in `src/vectorstore.py` and `src/ingest.py`
- **Adding new LLM provider**: Edit `src/llm.py` to add provider logic and update `app.py` sidebar options

## Project Structure

```
hcl_tech/
├── .env                  # Environment variables (not committed)
├── .env.example          # Template for .env
├── .gitignore           # Git ignore rules
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── src/
│   ├── __init__.py
│   ├── llm.py           # Provider factory and fallback logic
│   ├── qa.py            # QA pipeline: retrieve + answer
│   ├── ingest.py        # Document ingestion pipeline
│   ├── evaluate.py      # Evaluation script
│   └── vectorstore.py   # Shared ChromaDB helper (NEW)
├── data/
│   ├── raw/             # Uploaded/Raw documents
│   └── dictionary.csv   # Data dictionary
├── chroma_db/           # Persistent ChromaDB storage (generated)
├── results/             # Evaluation results and metrics
├── docs/                # Documentation
├── figures/             # Charts and plots
├── notebooks/           # Jupyter notebooks
├── Dataset*.csv         # Generated datasets
├── Project_*.docx     # Project documents
└── cleaned_dataset.csv
```