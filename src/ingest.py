#!/usr/bin/env python
"""Load PDF/DOCX/TXT files, split with RecursiveCharacterTextSplitter,
embed with sentence-transformers, store in persistent ChromaDB.

Features:
- Idempotent: resets collection before ingesting (stable chunk IDs)
- Chunk IDs: filename_page_chunknumber format
- Persistent ChromaDB storage
- Metadata: source file, page number, chunk_id
- Cosine similarity space
- Uses python-docx and PyPDF2 for file loading (no langchain loader dependencies)
- Functions take explicit paths, no module-level state
"""

import os
import shutil
from pathlib import Path

# Try to import PyPDF2, fall back if needed
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from docx import Document
    PYDOCX_AVAILABLE = True
except ImportError:
    PYDOCX_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

# Configuration constants (not module-level state used for control flow)
DATA_DIR_DEFAULT = Path("data/raw")
PERSIST_DIR_DEFAULT = Path("chroma_db")
COLLECTION_NAME_DEFAULT = "policy_docs"

# Chunking parameters
CHUNK_SIZE_DEFAULT = 500
CHUNK_OVERLAP_DEFAULT = 100

# Embedding model (local, no API key needed)
EMBEDDING_MODEL_NAME_DEFAULT = "all-MiniLM-L6-v2"


def read_pdf_file(filepath):
    """Read a PDF file using PyPDF2 and return list of (page_num, text) tuples."""
    texts = []
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text.strip():
                        texts.append((page_num, text))
                except Exception:
                    continue
        print(f"Read {len(texts)} pages from PDF: {filepath.name}")
        return texts
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        return []


def read_docx_file(filepath):
    """Read a DOCX file using python-docx and return list of (page_num, text) tuples."""
    texts = []
    try:
        doc = Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        # Treat the whole document as one "page" for simplicity
        combined = " ".join(full_text)
        if combined.strip():
            texts.append((1, combined))
        print(f"Read DOCX: {filepath.name}, {len(full_text)} paragraphs")
        return texts
    except Exception as e:
        print(f"Error reading DOCX {filepath}: {e}")
        return []


def read_txt_file(filepath):
    """Read a TXT file and return list of (page_num, text) tuples."""
    texts = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if content.strip():
            # Split into pages by double newlines
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs, 1):
                texts.append((i, para))
        print(f"Read TXT: {filepath.name}, {len(texts)} paragraphs")
        return texts
    except Exception as e:
        print(f"Error reading TXT {filepath}: {e}")
        return []


def get_chunk_id(filename, page, chunk_index):
    """Generate a stable chunk ID: filename_page_chunkindex."""
    base_name = Path(filename).stem
    return f"{base_name}_page{page}_chunk{chunk_index}"


def init_vectorstore(persist_dir=PERSIST_DIR_DEFAULT, collection_name=COLLECTION_NAME_DEFAULT):
    """Initialize the ChromaDB vectorstore with shared configuration.

    Uses the shared vectorstore helper to avoid signature mismatches.

    Args:
        persist_dir: Path to ChromaDB persist directory
        collection_name: ChromaDB collection name

    Returns:
        ChromaDB collection object
    """
    from src.vectorstore import get_collection
    return get_collection(path=persist_dir, collection_name=collection_name)


def ingest_directory(data_dir=None, persist_dir=None, collection_name=None):
    """Ingest all documents from a directory.

    Args:
        data_dir: Path to directory containing documents (default: data/raw)
        persist_dir: Path to ChromaDB persist directory (default: chroma_db)
        collection_name: ChromaDB collection name (default: policy_docs)

    Returns:
        int: Number of chunks created
    """
    data_dir = Path(data_dir or DATA_DIR_DEFAULT)
    persist_dir = Path(persist_dir or PERSIST_DIR_DEFAULT)
    collection_name = collection_name or COLLECTION_NAME_DEFAULT

    # Reset collection for idempotent ingestion
    print("\n[1] Resetting ChromaDB collection for idempotent ingestion...")
    try:
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
            print(f"Removed existing ChromaDB at {persist_dir}")
    except Exception as e:
        print(f"Note: Could not remove old DB: {e}")

    # Ensure persist directory exists
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Collect all document files
    print(f"\n[2] Scanning data directory: {data_dir}")
    all_files = []
    if data_dir.exists():
        for f in sorted(data_dir.iterdir()):
            if f.suffix.lower() in (".pdf", ".docx", ".txt"):
                all_files.append(f)
    print(f"Found {len(all_files)} document files: {[f.name for f in all_files]}")

    # Read all documents
    all_docs = []
    for filepath in all_files:
        print(f"\n  Reading {filepath.name}...")
        if filepath.suffix.lower() == ".pdf":
            if not PYPDF2_AVAILABLE:
                print(f"  SKIP: PyPDF2 not available")
                continue
            pages = read_pdf_file(filepath)
        elif filepath.suffix.lower() == ".docx":
            if not PYDOCX_AVAILABLE:
                print(f"  SKIP: python-docx not available")
                continue
            pages = read_docx_file(filepath)
        elif filepath.suffix.lower() == ".txt":
            pages = read_txt_file(filepath)
        else:
            continue

        for page_num, page_text in pages:
            chunk_id = get_chunk_id(filepath.name, page_num, 0)
            doc = LCDocument(
                page_content=page_text,
                metadata={
                    "source": filepath.name,
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "filepath": str(filepath),
                },
            )
            all_docs.append(doc)

    print(f"\nTotal documents read: {len(all_docs)}")

    # Split into chunks
    print(f"\n[3] Splitting documents (chunk_size={CHUNK_SIZE_DEFAULT}, overlap={CHUNK_OVERLAP_DEFAULT})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_DEFAULT, chunk_overlap=CHUNK_OVERLAP_DEFAULT
    )
    all_chunks = []
    for doc in all_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(chunks):
            chunk_id = get_chunk_id(
                doc.metadata["source"], doc.metadata["page"], i + 1
            )
            chunk_doc = LCDocument(
                page_content=chunk_text,
                metadata={
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"],
                    "chunk_id": chunk_id,
                    "original_chunk": doc.metadata["chunk_id"],
                },
            )
            all_chunks.append(chunk_doc)

    print(f"Total chunks: {len(all_chunks)}")

    # Embed and store in ChromaDB using shared vectorstore
    print(f"\n[4] Initializing embeddings model: {EMBEDDING_MODEL_NAME_DEFAULT}")
    collection = init_vectorstore(persist_dir=persist_dir, collection_name=collection_name)

    print(f"\n[5] Storing {len(all_chunks)} chunks in ChromaDB...")
    # Add documents directly using collection.add()
    for i, chunk_doc in enumerate(all_chunks):
        # Generate a unique ID for each chunk
        chunk_id = chunk_doc.metadata.get("chunk_id", f"chunk_{i}")
        collection.add(
            documents=[chunk_doc.page_content],
            metadatas=[chunk_doc.metadata],
            ids=[chunk_id],
        )

    print(f"ChromaDB persisted to {persist_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print(f"  - Files processed: {len(all_files)}")
    print(f"  - Documents read: {len(all_docs)}")
    print(f"  - Chunks created: {len(all_chunks)}")
    print(f"  - Vector store: {persist_dir}/collection")
    print(f"  - Embedding model: {EMBEDDING_MODEL_NAME_DEFAULT}")
    print("=" * 60)

    return len(all_chunks)


def ingest_files(file_paths=None, persist_dir=None, collection_name=None):
    """Ingest specific document files.

    Args:
        file_paths: List of file paths to ingest (default: scan data/raw)
        persist_dir: Path to ChromaDB persist directory (default: chroma_db)
        collection_name: ChromaDB collection name (default: policy_docs)

    Returns:
        int: Number of chunks created
    """
    data_dir = Path("data/raw")
    persist_dir = Path(persist_dir or PERSIST_DIR_DEFAULT)
    collection_name = collection_name or COLLECTION_NAME_DEFAULT

    # If specific files provided, copy them to data/raw first
    if file_paths:
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        for fp in file_paths:
            src = Path(fp)
            dst = data_dir / src.name
            if src.exists():
                shutil.copy2(str(src), str(dst))

    return ingest_directory(data_dir=data_dir, persist_dir=persist_dir,
                            collection_name=collection_name)


def main():
    """CLI entry point for ingestion."""
    print("=" * 60)
    print("RAG Document Q&A System - Ingestion Pipeline")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR_DEFAULT),
                        help="Directory containing documents (default: data/raw)")
    parser.add_argument("--persist-dir", type=str, default=str(PERSIST_DIR_DEFAULT),
                        help="ChromaDB persist directory (default: chroma_db)")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME_DEFAULT,
                        help="ChromaDB collection name (default: policy_docs)")
    parser.add_argument("--files", type=str, nargs="+",
                        help="Specific files to ingest (overrides data dir scan)")

    args = parser.parse_args()

    file_paths = args.files

    print(f"Ingesting from: {args.data_dir}")
    if file_paths:
        print(f"Specific files: {file_paths}")

    chunks_created = ingest_directory(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    if file_paths:
        # Also ingest the specific files
        chunks_created += ingest_files(
            file_paths=args.files,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
    )

    print("=" * 60)

    if __name__ == "__main__":
        main()