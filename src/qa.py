#!/usr/bin/env python
"""
src/qa.py

Retrieve top-k=4 chunks from ChromaDB, prompt the LLM, return answer + source chunks + retrieval scores + response time.

Features:
- Retrieve top-4 most similar chunks using cosine similarity
- Display similarity as 1 - distance (cosine)
- Prompt: "Answer ONLY from the context. If not found, say you don't know."
- Return answer, source chunks, retrieval scores, response time
- Similarity format: file (page X) — similarity: Y%
"""

import time
from pathlib import Path

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_K = 4  # Retrieve top-k=4 chunks


def get_chunk_id(filename, page, chunk_index):
    """Generate a stable chunk ID."""
    base_name = Path(filename).stem
    return f"{base_name}_page{page}_chunk{chunk_index}"


def retrieve_chunks(query, k=CHUNK_K, persist_dir="chroma_db", collection_name="policy_docs"):
    """
    Retrieve top-k similar chunks for a given query.

    Args:
        query: The user query string
        k: Number of chunks to retrieve (default 4)
        persist_dir: Path to ChromaDB persist directory
        collection_name: ChromaDB collection name

    Returns:
        list of dicts with keys: content, metadata, similarity_score, distance
    """
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Open existing ChromaDB
    db = Chroma(
        persist_directory=str(Path(persist_dir)),
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # Search for similar documents
    print(f"Retrieving top {k} chunks for query: {query[:80]}...")
    results = db.similarity_search_with_score(query, k=k)

    # Format results
    formatted_results = []
    for doc, score in results:
        metadata = doc.metadata
        # Cosine similarity: 1 - distance, clamped to [0, 1]
        similarity = max(0, min(1, 1 - score))
        # Format similarity as percentage
        similarity_pct = f"{similarity * 100:.1f}%"

        result = {
            "content": doc.page_content,
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", "unknown"),
            "chunk_id": metadata.get("chunk_id", "unknown"),
            "similarity": similarity,
            "similarity_pct": similarity_pct,
            "distance": score,
        }
        formatted_results.append(result)

    print(f"Retrieved {len(formatted_results)} chunks")
    return formatted_results


def generate_answer(query, context_chunks):
    """
    Generate an answer from the retrieved context chunks.

    Prompt: "Answer ONLY from the context. If not found, say you don't know."

    Args:
        query: The user's question
        context_chunks: List of retrieved chunk dicts (from retrieve_chunks)

    Returns:
        dict with keys: answer, response_time, source_info
    """
    start_time = time.time()

    if not context_chunks:
        response_time = time.time() - start_time
        return {
            "answer": "I don't know based on the provided documents.",
            "response_time": response_time,
            "source_info": "No relevant source chunks found.",
        }

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source_info = f"({chunk['source']}, page {chunk['page']})"
        context_parts.append(f"[Source {i}] {chunk['content']} {source_info}")

    context_text = "\n\n".join(context_parts)

    # Create prompt
    prompt_template = PromptTemplate(
        template="""Answer ONLY from the context below. If the answer is not found in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}""",
        input_variables=["context", "question"],
    )

    prompt = prompt_template.format(context=context_text, question=query)

    # Note: The actual LLM generation is handled in app.py/llm.py
    # This function returns the structured result for the UI to display
    response_time = time.time() - start_time

    # Format source info for display
    source_info_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        similarity_str = f"similarity: {chunk['similarity_pct']}"
        source_info_parts.append(
            f"{i}. {chunk['source']} (page {chunk['page']}) — {similarity_str}"
        )

    source_info = "\n".join(source_info_parts)

    return {
        "answer": f"[Answer generation handled by LLM provider - see app.py]",
        "response_time": response_time,
        "source_info": source_info,
        "context_text": context_text,
    }


def answer_from_context(query, k=CHUNK_K):
    """
    Full pipeline: retrieve chunks and format for answer display.

    Args:
        query: User's question
        k: Number of chunks to retrieve

    Returns:
        dict with: answer, source_chunks, retrieval_scores, response_time
    """
    start_time = time.time()

    # Step 1: Retrieve top-k chunks
    retrieved = retrieve_chunks(query, k=k)

    # Step 2: Generate response time
    retrieval_time = time.time() - start_time

    # Step 3: Format source chunks for display
    source_chunks = []
    for chunk in retrieved:
        source_chunks.append({
            "file": chunk["source"],
            "page": chunk["page"],
            "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            "similarity": chunk["similarity_pct"],
        })

    # Step 4: Format retrieval scores
    retrieval_scores = [chunk["similarity_pct"] for chunk in retrieved]

    # Step 5: Construct answer (retrieval-only mode answer)
    if not retrieved:
        answer = "I don't know based on the provided documents."
    else:
        # Use the first chunk's content as a basic answer placeholder,
        # or combine key information
        answers = []
        for chunk in retrieved:
            # Extract potential answer from context
            content = chunk["content"]
            # If the question starts with who/what/where, try to find matching text
            answers.append(content[:300] + "..." if len(content) > 300 else content)

        if answers:
            answer = "Based on the retrieved documents:\n\n" + "\n\n".join(
                [f"From {chunk['file']} (page {chunk['page']}) [{chunk['similarity']}]: {ans}"
                 for chunk, ans in zip(retrieved, answers)]
            )
        else:
            answer = "I don't know based on the provided documents."

    return {
        "answer": answer,
        "source_chunks": source_chunks,
        "retrieval_scores": retrieval_scores,
        "response_time": retrieval_time,
    }


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS FOR EVALUATION
# ---------------------------------------------------------------------------


def compute_hit_rate(retrieved_results, question_info):
    """
    Compute whether the expected source file appears in the top-4 retrieved chunks
    AND at least one key phrase appears in the retrieved text.

    Args:
        retrieved_results: List of dicts from retrieve_chunks (with 'source' and 'content')
        question_info: Dict with 'expected_source' and 'key_phrases'

    Returns:
        bool: True if hit criteria met, False otherwise
    """
    if not retrieved_results:
        return False

    expected_source = question_info.get("expected_source")
    key_phrases = question_info.get("key_phrases", [])

    # Check if expected source is in top-4
    source_in_top4 = False
    for result in retrieved_results[:4]:
        if expected_source and expected_source in result.get("source", ""):
            source_in_top4 = True
            break

    # Check if any key phrase appears in retrieved text
    key_phrase_found = False
    all_retrieved_text = " ".join(
        [result.get("content", "").lower() for result in retrieved_results]
    )
    for phrase in key_phrases:
        if phrase.lower() in all_retrieved_text:
            key_phrase_found = True
            break

    # Hit: expected source in top-4 AND key phrase in retrieved text
    hit = source_in_top4 and key_phrase_found
    return hit


def compute_answer_accuracy(retrieved_results, question_info, llm_available=False):
    """
    Compute answer accuracy based on keyword matching.

    Only computed when LLM is available. When no LLM key, returns None and states
    that accuracy cannot be computed.

    Args:
        retrieved_results: List of dicts from retrieve_chunks
        question_info: Dict with the evaluation question info
        llm_available: Boolean indicating if an LLM is available

    Returns:
        float or None: Accuracy score (0-1) if LLM available, None otherwise
    """
    if not llm_available:
        return None

    expected_source = question_info.get("expected_source")
    key_phrases = question_info.get("key_phrases", [])

    if not retrieved_results:
        return 0.0

    # Get the top retrieved chunk's content
    top_chunk = retrieved_results[0]
    retrieved_text = top_chunk.get("content", "").lower()

    # Check how many key phrases appear in the top answer
    if not key_phrases:
        return 0.5  # Neutral accuracy when no key phrases defined

    found_phrases = 0
    for phrase in key_phrases:
        if phrase.lower() in retrieved_text:
            found_phrases += 1

    accuracy = found_phrases / len(key_phrases) if key_phrases else 0.0
    return round(accuracy, 2)
