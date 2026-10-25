#!/usr/bin/env python
"""Shared vector store helper for ChromaDB usage.

Provides a single point of configuration for the ChromaDB persistent client
and embedding function, so all modules use the same interface and avoid
deprecation/signature mismatch errors.
"""

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ----------------------------------------------------------------------
# Shared ChromaDB persistent client
# ----------------------------------------------------------------------


def get_persistent_client(path="chroma_db"):
    """Get a persistent ChromaDB client instance.

    Args:
        path: Path to the persist directory (default: "chroma_db")

    Returns:
        ChromaDB PersistentClient instance
    """
    return chromadb.PersistentClient(path=str(path))


# ----------------------------------------------------------------------
# Shared embedding function using SentenceTransformers
# ----------------------------------------------------------------------


def get_sentence_transformer_embedding_function(model_name="all-MiniLM-L6-v2"):
    """Get a ChromaDB-compatible embedding function using SentenceTransformers.

    This is the recommended way to use SentenceTransformers embeddings with
    ChromaDB, avoiding the LangChain embedding function signature mismatch.

    Args:
        model_name: Name of the SentenceTransformers model (default:
            "all-MiniLM-L6-v2")

    Returns:
        ChromaDB EmbeddingFunction instance
    """
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        # Cosine similarity space - default for ChromaDB
    )


# ----------------------------------------------------------------------
# Shared collection getter
# ----------------------------------------------------------------------


def get_collection(
    path="chroma_db",
    collection_name="policy_docs",
    embedding_function=None,
    space="cosine",
):
    """Get or create a ChromaDB collection with consistent configuration.

    Args:
        path: Path to persist directory (default: "chroma_db")
        collection_name: Name of the collection (default: "policy_docs")
        embedding_function: Embedding function to use (default:
            SentenceTransformerEmbeddingFunction with all-MiniLM-L6-v2)
        space: HNSW space for similarity search (default: "cosine")

    Returns:
        ChromaDB collection object
    """
    if embedding_function is None:
        embedding_function = get_sentence_transformer_embedding_function()

    client = get_persistent_client(path)

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": space},
    )


# ----------------------------------------------------------------------
# Convenience: pre-configured collection for this project
# ----------------------------------------------------------------------


# Pre-configured collection instance - used by all modules
POLICY_DOCS_COLLECTION = None


def init_collection():
    """Initialize the pre-configured policy_docs collection.

    Should be called once at app startup. Creates the collection with
    the standard SentenceTransformer embedding function and cosine similarity.
    """
    global POLICY_DOCS_COLLECTION
    POLICY_DOCS_COLLECTION = get_collection(
        path="chroma_db",
        collection_name="policy_docs",
    )
    return POLICY_DOCS_COLLECTION


def get_policy_docs_collection():
    """Get the pre-configured policy_docs collection.

    Returns the globally initialized collection. Call init_collection()
    first if not already initialized.
    """
    global POLICY_DOCS_COLLECTION
    if POLICY_DOCS_COLLECTION is None:
        POLICY_DOCS_COLLECTION = init_collection()
    return POLICY_DOCS_COLLECTION