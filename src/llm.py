#!/usr/bin/env python
"""
src/llm.py

Provider factory and fallback logic for LLM calls.

Supported providers (switchable via .env LLM_PROVIDER):
- gemini: using langchain-google-genai (requires GEMINI_API_KEY)
- ollama: using langchain-ollama (model running locally via Ollama)
- retrieval-only: when no API key is available, show source chunks only

Fallback logic when GEMINI_API_KEY is empty:
1. Check if Ollama is running (http://localhost:11434/api/tags)
2. If Ollama available, use ollama model
3. If not, enter retrieval-only mode

All LLM calls are wrapped in try/except to handle:
- Invalid API key
- Quota exceeded
- Model not found
- Network errors
"""

import os
import json
import time
import socket
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


def check_ollama_running():
    """Check if Ollama is running and available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_active_provider(
    llm_provider=None,
    gemini_api_key=None,
    gemini_model=None,
    openai_api_key=None,
    openai_model=None,
    ollama_model=None,
):
    """
    Determine the active LLM provider and model based on available keys.

    This is a pure function: no module-level state is modified.

    Order of precedence:
    1. Configured provider with a valid key
    2. Ollama if running locally
    3. Retrieval-only mode

    Args:
        llm_provider: str or None - overrides LLM_PROVIDER env var
        gemini_api_key: str or None - overrides GEMINI_API_KEY env var
        gemini_model: str or None - overrides GEMINI_MODEL env var
        openai_api_key: str or None - overrides OPENAI_API_KEY env var
        openai_model: str or None - overrides OPENAI_MODEL env var
        ollama_model: str or None - overrides OLLAMA_MODEL env var

    Returns:
        tuple of (provider, model, key_status)
    """
    # Use provided args or fall back to env vars
    provider = (llm_provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    key = (gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    model = (gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).lower()
    openai_key = openai_api_key or os.getenv("OPENAI_API_KEY", "").strip()
    openai_model_name = (openai_model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")).lower()
    ollama = (ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2")).lower()

    # 1. Check configured provider with valid key
    if provider == "gemini" and key:
        return "gemini", model, "green"
    elif provider == "openai" and openai_key:
        return "openai", openai_model_name, "green"
    elif provider == "ollama":
        # Check if Ollama is running
        if check_ollama_running():
            return "ollama", ollama, "green"
        # Fall through to retrieval-only if Ollama not running

    # 2. Fall back to Ollama if running
    if check_ollama_running():
        return "ollama", ollama, "yellow"

    # 3. Retrieval-only mode
    return "retrieval-only", "retrieval-only", "red"


def generate_answer(question, context_chunks, llm_provider=None, gemini_api_key=None,
                    gemini_model=None, openai_api_key=None, openai_model=None,
                    ollama_model=None):
    """
    Generate an answer using the active LLM provider.

    This is a pure function: get_active_provider() is called internally.
    No module-level state is required.

    Args:
        question: The user's question
        context_chunks: List of retrieved text chunks
        llm_provider, gemini_api_key, etc.: Optional overrides for provider config

    Returns:
        dict with 'answer', 'provider', 'model', 'response_time', 'source_info'
    """
    from langchain_core.prompts import PromptTemplate

    # Auto-detect provider - pure function call, no module state needed
    active_provider, active_model, key_status = get_active_provider(
        llm_provider=llm_provider,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        ollama_model=ollama_model,
    )

    start_time = time.time()

    # Build the context from retrieved chunks
    context_text = "\n\n".join(
        [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)]
    )

    # Prompt: "Answer ONLY from the context. If not found, say you don't know."
    prompt_template = PromptTemplate(
        template="""Answer ONLY from the context below. If the answer is not found in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}""",
        input_variables=["context", "question"],
    )

    prompt = prompt_template.format(context=context_text, question=question)

    # Try the active provider
    if active_provider == "gemini":
        return _generate_gemini(prompt, start_time, active_model)
    elif active_provider == "ollama":
        return _generate_ollama(prompt, start_time, active_model)
    elif active_provider == "openai":
        return _generate_openai(prompt, start_time, active_model)
    else:
        # Retrieval-only mode
        response_time = time.time() - start_time
        return {
            "answer": "I don't know based on the provided documents.",
            "provider": "retrieval-only",
            "model": "retrieval-only",
            "response_time": response_time,
            "source_info": _format_source_chunks(context_chunks),
        }


def _generate_gemini(prompt, start_time, model_name):
    """Generate answer using Gemini API."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY", ""),
            temperature=0.7,
        )
        response = llm.invoke(prompt)
        response_time = time.time() - start_time
        return {
            "answer": response.content if hasattr(response, 'content') else str(response),
            "provider": "gemini",
            "model": model_name,
            "response_time": response_time,
            "source_info": "",
        }
    except Exception as e:
        # If Gemini fails, fall back to Ollama or retrieval-only
        print(f"Gemini error: {e}")
        from .llm import check_ollama_running
        if check_ollama_running():
            # Re-call generate_answer with ollama fallback via retrieval-only result
            return {
                "answer": f"Error: Gemini API request failed. {str(e)[:200]}. Falling back to retrieval-only mode.",
                "provider": "retrieval-only",
                "model": "retrieval-only",
                "response_time": time.time() - start_time,
                "source_info": _format_source_chunks([]),
            }
        return {
            "answer": f"Error: Gemini API request failed. {str(e)[:200]}. System running in retrieval-only mode.",
            "provider": "retrieval-only",
            "model": "retrieval-only",
            "response_time": time.time() - start_time,
            "source_info": _format_source_chunks([]),
        }


def _generate_ollama(prompt, start_time, model_name):
    """Generate answer using Ollama API."""
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            base_url="http://localhost:11434",
        )
        response = llm.invoke(prompt)
        response_time = time.time() - start_time
        return {
            "answer": response.content if hasattr(response, 'content') else str(response),
            "provider": "ollama",
            "model": model_name,
            "response_time": response_time,
            "source_info": "",
        }
    except Exception as e:
        print(f"Ollama error: {e}")
        # Fall back to retrieval-only
        return {
            "answer": f"Error: Ollama request failed. {str(e)[:200]}. Running in retrieval-only mode.",
            "provider": "retrieval-only",
            "model": "retrieval-only",
            "response_time": time.time() - start_time,
            "source_info": _format_source_chunks([]),
        }


def _generate_openai(prompt, start_time, model_name):
    """Generate answer using OpenAI API."""
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        response = llm.invoke(prompt)
        response_time = time.time() - start_time
        return {
            "answer": response.content if hasattr(response, 'content') else str(response),
            "provider": "openai",
            "model": model_name,
            "response_time": response_time,
            "source_info": "",
        }
    except Exception as e:
        print(f"OpenAI error: {e}")
        return {
            "answer": f"Error: OpenAI request failed. {str(e)[:200]}.",
            "provider": "retrieval-only",
            "model": "retrieval-only",
            "response_time": time.time() - start_time,
            "source_info": _format_source_chunks([]),
        }


def _format_source_chunks(chunks):
    """Format source chunks for display."""
    if not chunks:
        return "No source chunks available."
    formatted = []
    for i, chunk in enumerate(chunks[:4], 1):
        formatted.append(f"{i}. {chunk[:100]}...")
    return "\n".join(formatted)


def get_provider_status(llm_provider=None, gemini_api_key=None,
                        gemini_model=None, openai_api_key=None, openai_model=None,
                        ollama_model=None):
    """
    Return the current provider status for display in sidebar.

    Pure function: calls get_active_provider() internally.
    No module-level state is required.
    """
    active_provider, active_model, key_status = get_active_provider(
        llm_provider=llm_provider,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        ollama_model=ollama_model,
    )
    return {
        "provider": active_provider,
        "model": active_model,
        "key_status": key_status,
        "gemini_key_present": key_status != "red",
    }