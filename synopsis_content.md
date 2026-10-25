# Project Synopsis Content

## 1. Title
RAG-Based Document Q&A System

## 2. Project Overview
A Retrieval-Augmented Generation (RAG) system that enables AI-based question-answering over policy/guideline documents (PDF, DOCX, TXT). The system ingests documents, creates local embeddings using sentence-transformers, stores them in ChromaDB, and retrieves relevant context for user queries. Answers can be generated using switchable LLM providers (gemini, ollama, openai) or displayed in retrieval-only mode when no API key is available.

Use case: AI-based Q&A over policy/guideline documents for college industry projects.

## 3. Problem Statement
Organizations struggle with making their internal policy documents searchable. Employees cannot find relevant information quickly, leading to repeated questions, policy violations, and inefficiency. This project develops a RAG system that ingests various document formats, creates vector embeddings, and provides natural language query capability over the document corpus.

## 4. Objectives
- Build a complete RAG pipeline: ingestion → embedding → storage → retrieval → answer generation
- Support multiple document formats: PDF, DOCX, TXT
- Use local, free embeddings (all-MiniLM-L6-v2, no API key needed)
- Implement LLM provider fallback: gemini → ollama → retrieval-only
- Create a Streamlit UI with source chunk verification
- Run evaluation: retrieval hit rate, response time, answer accuracy
- Generate comprehensive documentation and reports

## 5. Scope
### In-Scope
- Document ingestion and chunking (chunk_size=500, overlap=100)
- Local embeddings with sentence-transformers
- ChromaDB vector storage with cosine similarity
- Retrieval of top-k=4 chunks with similarity scores
- LLM provider factory with fallback logic
- Streamlit UI with chat interface and source displays
- Evaluation: 15 in-scope + 3 out-of-scope questions
- Metrics: retrieval hit rate, average response time, answer accuracy
- EDA: 4 visualizations, 5 written insights
- Documentation: report, synopsis, demo script, architecture

### Out-of-Scope
- Production deployment with custom domains
- Multi-user authentication and authorization
- Real-time document updates without restart
- Advanced re-ranking or query expansion
- Integration with external databases or APIs (beyond configured LLM providers)
- Mobile application development
- Real-time collaboration features

## 6. Methodology
### 6.1 Data Preparation
- Created 7 policy documents: 3 PDF, 2 DOCX, 2 TXT
- Each document: 1-2 pages of realistic college-industry policy content
- Generated dataset.csv: 500 rows with columns doc_id, doc_type, chunk_id, word_count, retrieval_score, query_type, answer_quality_score
- Filled data_dictionary.csv: 7 columns with descriptions

### 6.2 Ingestion Pipeline
- Load PDF/DOCX/TXT using PyPDF2 and python-docx
- Generate stable chunk IDs: filename_page_chunknumber
- RecursiveCharacterTextSplitter: chunk_size=500, chunk_overlap=100
- HuggingFaceEmbeddings: all-MiniLM-L6-v2
- ChromaDB: persistent storage, cosine similarity space
- Idempotent: collection reset before ingest

### 6.3 LLM Provider Factory
- Read LLM_PROVIDER from .env (default: gemini)
- Check GEMINI_API_KEY presence
- Fallback: check Ollama at localhost:11434
- If no Ollama: retrieval-only mode
- Wrap all LLM calls in try/except
- Friendly UI error messages instead of tracebacks

### 6.4 QA Pipeline
- Retrieve top-k=4 chunks from ChromaDB
- Prompt: "Answer ONLY from the context. If not found, say you don't know."
- Return: answer + source chunks + retrieval scores + response time
- Similarity format: file (page X) — similarity: Y%

### 6.5 Streamlit UI
- st.cache_resource for embedding model and vector store
- st.session_state for chat history
- Sidebar: provider selection, model name, key status, file upload
- Main area: chat box, query input, answer with expandable source chunks
- Warning banners: hallucination warning + retrieval-only banner

### 6.6 Evaluation
- 15 in-scope questions from sample documents
- 3 out-of-scope questions (test "I don't know" behavior)
- Retrieval hit rate: expected source in top-4 AND key phrase in text
- Average response time per question
- Answer accuracy: keyword match, only when LLM available
- Save results to results/metrics.csv

### 6.7 EDA
- Load dataset.csv, clean and check (nulls, duplicates, dtypes)
- Save cleaned_dataset.csv
- 4 visualizations: retrieval_score by doc_type, answer_quality by query_type, word_count histogram, correlation heatmap
- 5 written insights under each chart

### 6.8 Documentation
- report/Project_Report.md: 8+ pages with all sections
- synopsis_content.md: text for every section of the Project_Synopsis_Template
- demo_script.md: 90-second demo video script
- docs/architecture.md: Mermaid architecture diagram + Level-0 DFD

## 7. Results
### 7.1 Retrieval Metrics
- Total questions: 18 (15 in-scope + 3 out-of-scope)
- Retrieval hit rate: 13/15 in-scope questions had expected source in top-4 with key phrases
- Average response time: 2.43 seconds

### 7.2 EDA Findings
1. Document type affects retrieval (TXT > PDF)
2. Query type impacts quality (analytical > summarization)
3. Average document length: 268.8 words
4. Near-zero correlation between retrieval scores and answer quality (r = -0.005, p = 0.92)
5. Query distribution: summarization most common (118), analytical least common (90)

### 7.3 System Metrics
- Active provider: retrieval-only (no API key)
- Key status: red
- System mode: retrieval-only (source chunks displayed, no generated answers)
- Requirements: Python 3.11+, ~200MB for embedding model

## 8. Future Enhancements
- Ollama integration for local LLM fallback
- Cross-encoder re-ranking for improved answer quality
- Multi-document spanning queries
- User feedback system (thumbs-up/down)
- Cloudflare Sites deployment with D1/R2
- Additional document format support
- Hybrid search (keyword + vector)

## 9. Ethical Considerations
- Hallucination risk mitigated by source chunk display
- Data privacy: local processing, no external upload without key
- API key security: .env excluded from git, never share in chat
- Use case limitations: not for legal or medical advice
- Bias awareness: embedding model may have training data biases
- Disclaimer: verify with source documents, not for legal/medical decisions

## 10. Appendices
### 10.1 AI Tool Usage
This project used AI assistance for code generation, documentation writing, EDA visualization, and research. All code was reviewed, tested, and verified to work correctly. AI-generated code was adapted to meet all project requirements including fallback logic, error handling, and Streamlit UI integration.

### 10.2 Technical Decisions
- Local embeddings ensure zero-cost operation
- Cosine similarity with 1-distance display for user-friendly scores
- Idempotent ingestion via stable chunk IDs
- Three-tier fallback: gemini → ollama → retrieval-only
- Source verification at every answer to combat hallucination

### 10.3 Installation Commands
```bash
# Install requirements
pip install -r requirements.txt

# Run ingestion
python src/ingest.py

# Run evaluation
python src/evaluate.py

# Run EDA
python notebooks/eda.py

# Start Streamlit
streamlit run app.py

# Get Gemini key at: aistudio.google.com
```