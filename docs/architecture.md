# System Architecture

## 1. High-Level Architecture Diagram

The following Mermaid diagram illustrates the system architecture:

```mermaid
graph TD
    % Client Layer
    user([User]) -->|HTTP Requests| streamlit([Streamlit UI])
    streamlit -->|File Uploads| filesystem([File System])
    streamlit -->|API Configuration| env([.env File])

    % Application Layer
    streamlit -->|Ingest Documents| ingest([ingest.py])
    streamlit -->|User Queries| qa([qa.py])
    streamlit -->|Run Evaluation| evaluate([evaluate.py])

    % Data & AI Layer
    ingest -->|Load & Chunk| docs[(Raw Documents)]
    ingest -->|Embed & Store| chromadb[(ChromaDB)]
    qa -->|Retrieve Context| chromadb
    qa -->|Generate Answer| llm[(LLM Provider)]
    evaluate -->|Load Dataset| dataset[(dataset.csv)]

    % LLM Provider Layer
    llm -->|gemini| google([Google Gemini API])
    llm -->|ollama| ollama_local([Ollama Local])
    llm -->|retrieval-only| local_mode([Retrieval-Only Mode])

    % Style Configurations
    style streamlit fill:#bbf,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style chromadb fill:#f9f,stroke:#333,stroke-width:2px
    style env fill:#ff9,stroke:#333,stroke-width:2px
    style llm fill:#cfc,stroke:#333,stroke-width:2px
    style user fill:#ffc,stroke:#333,stroke-width:2px
```

## 2. Component Details

### 2.1 Streamlit UI (`app.py`)
- **Role**: User-facing frontend application
- **Inputs**: 
  - Query text input (chat_input)
  - Provider selection (sidebar radio: gemini/ollama/retrieval-only)
  - API key status (via .env configuration)
  - File uploads (sidebar file_uploader)
- **Outputs**:
  - Chat message history (st.session_state.chat_history)
  - Answer display with source chunks (expandable expander)
  - Warning banners (hallucination warning + retrieval-only banner)
  - Sidebar controls (provider, model, key status, file upload)

### 2.2 Ingestion Pipeline (`src/ingest.py`)
- **Role**: Document processing and vector store population
- **Inputs**:
  - Raw policy documents (data/raw/: PDF, DOCX, TXT)
  - Chunking parameters (chunk_size=500, overlap=100)
- **Processes**:
  1. Load files: PyPDF2 for PDFs, python-docx for DOCX, native read for TXT
  2. Text splitting: RecursiveCharacterTextSplitter
  3. Embedding: HuggingFaceEmbeddings (all-MiniLM-L6-v2)
  4. Vector storage: ChromaDB with cosine similarity
  5. Metadata: source file, page number, chunk_id
- **Outputs**:
  - Persistent ChromaDB storage (chroma_db/)
  - Idempotent collection reset on re-runs

### 2.3 LLM Provider Factory (`src/llm.py`)
- **Role**: Determine and manage the active LLM provider
- **Inputs**:
  - LLM_PROVIDER from .env (default: gemini)
  - GEMINI_API_KEY (may be empty)
  - OLLAMA_MODEL from .env
  - OPENAI_API_KEY from .env
- **Processes**:
  1. Read provider configuration
  2. Validate key presence
  3. Fallback logic:
     - gemini + key → use Gemini
     - gemini + no key → check Ollama → retrieval-only
     - ollama + running → use Ollama
     - ollama + not running → retrieval-only
     - openai + key → use OpenAI
     - openai + no key → retrieval-only
  4. Wrap all LLM calls in try/except
  5. Report status to UI sidebar
- **Outputs**:
  - Active provider name (gemini/ollama/retrieval-only)
  - Model name display
  - Key status (green/yellow/red)
  - Generated answers or fallback messages

### 2.4 QA Pipeline (`src/qa.py`)
- **Role**: Retrieve relevant context and format for answering
- **Inputs**:
  - User query string
  - k=4 (number of chunks to retrieve)
- **Processes**:
  1. Retrieve top-4 chunks from ChromaDB using similarity_search_with_score
  2. Compute similarity = 1 - cosine_distance
  3. Format similarity as percentage for display
  4. Build prompt: "Answer ONLY from the context. If not found, say you don't know."
  5. Return structure: answer + source_chunks + retrieval_scores + response_time
- **Outputs**:
  - List of retrieved chunks with: content, source, page, chunk_id, similarity, similarity_pct
  - Formatted answer string
  - Source chunk display format: "file (page X) — similarity: Y%"

### 2.5 Evaluation Module (`src/evaluate.py`)
- **Role**: System evaluation and metrics computation
- **Inputs**:
  - 15 in-scope questions from policy documents
  - 3 out-of-scope questions (general knowledge)
  - Provider status from src.llm
- **Processes**:
  1. For each question, retrieve top-4 chunks
  2. Compute hit rate: expected source in top-4 AND key phrase in text
  3. Compute average response time
  4. Compute answer accuracy (keyword match, LLM-available only)
  5. Save results to results/metrics.csv
  6. Save results to results/evaluation_results.json
- **Outputs**:
  - Per-question results with hit/accuracy flags
  - Summary metrics: hit rate, avg response time, overall accuracy
  - JSON summary for programmatic access

### 2.6 ChromaDB Vector Store
- **Role**: Persistent vector similarity search
- **Configuration**:
  - Embedding function: all-MiniLM-L6-v2 (HuggingFace)
  - Collection name: policy_docs
  - Similarity space: cosine (hnsw_space: "cosine")
  - Persist directory: chroma_db/
  - Automatic document persistence (no manual persist() needed in v0.4.x)
- **Metadata per chunk**:
  - source: filename (e.g., it_security_policy.pdf)
  - page: page number (1, 2, etc.)
  - chunk_id: stable ID (filename_page_chunkN)
- **Queries**:
  - similarity_search_with_score(query, k=4)
  - Returns: (Document, distance) pairs
  - Distance is cosine distance; similarity = 1 - distance

### 2.7 .env Configuration
- **Role**: System configuration without committing secrets to version control
- **Contents**:
  - LLM_PROVIDER=gemini (switchable: gemini | ollama | openai)
  - GEMINI_API_KEY= (empty by default; paste key from aistudio.google.com)
  - GEMINI_MODEL=gemini-2.5-flash
  - OPENAI_API_KEY= (optional)
  - OLLAMA_MODEL=llama3.2 (optional, requires local Ollama)
- **Security**:
  - .env excluded from git (.gitignore)
  - Never commit real API keys
  - .env.example shows format without values

## 3. Data Flow Diagrams

### 3.1 Level-0 DFD (Data-Flow Diagram)

```mermaid
graph TD
    % External Entities
    user([User]) -->|Queries| streamlit([Streamlit UI])
    streamlit -->|Document Files| filesystem([File System])
    streamlit -->|API Keys| env([.env File])

    % Processes
    streamlit -->|Ingest Documents| ingest([ingest.py])
    streamlit -->|Submit Query| qa([qa.py])
    streamlit -->|Run Evaluation| evaluate([evaluate.py])
    qa -->|Retrieve Context| chromadb[(ChromaDB)]
    evaluate -->|Load Dataset| dataset[(dataset.csv)]

    % Data Stores
    chromadb -->|Store Embeddings| embeddings[(all-MiniLM-L6-v2)]
    ingest -->|Write ChromaDB| chromadb
    evaluate -->|Read/Write| cleaned_dataset[(cleaned_dataset.csv)]

    % Responses
    chromadb -->|Similarity Results| qa
    qa -->|Answer + Source Chunks| streamlit
    evaluate -->|Metrics CSV| results[(results/metrics.csv)]
    evaluate -->|EDA Insights| insights[(results/eda_insights.txt)]

    %% Styling
    style user fill:#e6f7ff,stroke:#333,stroke-width:2px
    style streamlit fill:#ffe6e6,stroke:#333,stroke-width:2px
    style ingest fill:#ccffcc,stroke:#333,stroke-width:2px
    style qa fill:#ccccff,stroke:#333,stroke-width:2px
    style evaluate fill:#ffffcc,stroke:#333,stroke-width:2px
    style chromadb fill:#e6ffe6,stroke:#333,stroke-width:2px
    style dataset fill:#f0f8ff,stroke:#333,stroke-width:2px
    style results fill:#fff8dc,stroke:#333,stroke-width:2px
    style insights fill:#ffe6e6,stroke:#333,stroke-width:2px
```

### 3.2 Data Flow Descriptions

#### 3.2.1 Document Ingestion Flow
1. User uploads document files (PDF, DOCX, TXT) via Streamlit sidebar
2. `ingest.py` loads each file and extracts text:
   - PyPDF2 extracts text page-by-page from PDFs
   - python-docx extracts text from DOCX paragraphs
   - Native file reading for TXT files
3. Text is split using RecursiveCharacterTextSplitter (chunk_size=500, overlap=100)
4. Each chunk is embedded using sentence-transformers all-MiniLM-L6-v2
5. Embeddings with metadata (source, page, chunk_id) are stored in ChromaDB
6. ChromaDB persists to chroma_db/ directory for future queries

#### 3.2.2 Query/Answer Flow
1. User enters a query in the Streamlit chat input
2. `qa.py` retrieves top-4 similar chunks from ChromaDB using cosine similarity
3. Similarity scores are computed as `1 - distance`, displayed as percentage
4. Prompt is constructed: "Answer ONLY from the context. If not found, say you don't know."
5. If LLM provider is available (gemini/ollama):
   - LLM generates answer based on retrieved context
   - Answer and source chunks are displayed in UI
6. If no LLM provider (retrieval-only mode):
   - System displays "I don't know based on the provided documents"
   - Source chunks are still displayed with file, page, and similarity
   - Retrieval-only banner shows: "Gemini API key missing..."

#### 3.2.3 Evaluation Flow
1. `evaluate.py` loads dataset.csv (500 Q&A pairs)
2. For each evaluation question:
   - Retrieve top-4 chunks from ChromaDB
   - Check if expected source file is in top-4
   - Check if key phrases appear in retrieved text
   - Measure response time
   - If LLM available: compute answer accuracy via keyword matching
3. Aggregate metrics:
   - Retrieval hit rate: % of questions with expected source in top-4 + key phrases
   - Average response time: mean across all questions
   - Answer accuracy: % of questions with correct keyword matches (LLM only)
4. Save results:
   - results/metrics.csv: per-question detailed results
   - results/evaluation_results.json: summary metrics
   - results/eda_insights.txt: 5 written EDA insights

#### 3.2.3 Configuration Flow
1. System reads .env file at startup using python-dotenv
2. `get_provider_status()` determines:
   - Active LLM provider (gemini/ollama/retrieval-only)
   - Model name display
   - Key status (green: key present, yellow: Ollama available, red: no key)
3. Sidebar displays current provider and status
4. User can change provider in sidebar; system reconfigures on rerun
5. File uploads trigger re-ingestion with updated document set

## 4. Initialization Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit App
    participant I as ingest.py
    participant L as llm.py
    participant C as ChromaDB
    participant Q as qa.py

    U->>S: Open streamlit run app.py
    S->>I: Python import (on first load)
    I->>C: Load embeddings & open DB
    S->>L: get_provider_status()
    L->>L: Read .env, check keys, determine provider
    alt GEMINI_API_KEY present
        L-->>S: green status, gemini provider
    alt GEMINI_API_KEY empty
        L->>L: check_ollama_running()
        alt Ollama running
            L-->>S: yellow status, ollama provider
        alt Ollama not running
            L-->>S: red status, retrieval-only
        end
    end
    S->>S: Render UI with sidebar, chat area, banners
    U->>S: Enter query + press Enter
    S->>Q: retrieve_chunks(query, k=4)
    Q->>C: similarity_search_with_score(query, k=4)
    C-->>Q: (chunk, distance) pairs
    Q->>Q: format similarity = 1-distance, as percentage
    alt LLM available (gemini/ollama)
        Q->>L: generate_answer(query, context_chunks)
        L-->>Q: answer + source_info
    else Retrieval-only
        Q-->>Q: display "I don't know" + source chunks
    end
    Q-->>S: answer + source_chunks + scores + response_time
    S->>S: Display answer in chat_message(assistant)
    S->>S: Show expander with source chunks
    S->>S: Update chat_history
    S->>S: Rerun for next interaction
```

## 5. Deployment Configuration

### 5.1 Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize ChromaDB and ingest documents
python src/ingest.py

# 3. (Optional) Run EDA for insights
python notebooks/eda.py

# 4. Start the application
streamlit run app.py

# 5. To add Gemini key:
#   - Get key at: aistudio.google.com
#   - Add GEMINI_API_KEY=your_key_here to .env (no quotes or spaces)
#   - Restart: streamlit run app.py
```

### 5.2 Cloudflare Sites Deployment
```bash
# 1. First prepare Next.js if applicable
# sites_prepare_next with realtime flag if needed

# 2. Add storage for database and files
# sites_add_storage(siteId, environment="preview", kind="d1", name="DB")
# sites_add_storage(siteId, environment="preview", kind="r2", name="FILES")

# 3. Enable authentication if needed
# sites_enable_auth(siteId, environment="preview", localOrigin="http://localhost:8787")

# 4. Deploy
# sites_deploy(siteId, environment="preview", manifestPath="manifest.json")
```

### 5.3 Environment Variables for Production
- OPENAI_API_KEY or GEMINI_API_KEY set in Cloudflare Secrets
- D1 database binding: env.DB
- R2 bucket binding: env.FILES
- ALL_other reserved variables: FREEBUFF_CLIENT_ID, FREEBUFF_ISSUER, APP_ORIGIN

## 6. Technical Requirements

### 6.1 Python Dependencies
- langchain-core >= 1.7.0
- langchain >= 1.4.7
- langchain-community >= 0.4.2
- langchain-text-splitters >= 1.1.3
- langchain-huggingface >= 1.2.3
- sentence-transformers >= 2.2.0
- chromadb >= 1.5.0
- PyPDF2 >= 3.0.1
- python-docx >= 1.2.0
- streamlit >= 1.38.0
- python-dotenv >= 1.0.0

### 6.2 Hardware Requirements
- CPU: Any modern processor (embedding computation ~2-5 seconds for 43 chunks)
- RAM: 4GB minimum (8GB recommended for larger document sets)
- Storage: 200MB for embedding model + ChromaDB persistence

### 6.3 Software Requirements
- Python 3.11 or 3.12
- pip for package installation
- Git for version control (optional, for project management)

### 6.4 Optional Enhancements
- Ollama installed and running for local LLM fallback
- Cloudflare account for Sites deployment
- GEMINI_API_KEY from AI Studio for generated answers
- Additional document formats (XLSX, PPTX expansion)