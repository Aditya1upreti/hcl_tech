#!/usr/bin/env python
"""
src/evaluate.py

Evaluation script for the RAG Document Q&A System.

Features:
- 15 in-scope questions drawn from the sample policy documents
- 3 out-of-scope questions to test "I don't know" behaviour
- Metrics computed:
  - Retrieval hit rate: expected source file in top-4 AND key phrase in retrieved text
  - Average response time: per question
  - Answer accuracy: keyword match, computed ONLY when an LLM is available
- When no LLM key: report retrieval hit rate and average response time only,
  state clearly that answer accuracy cannot be computed
- Save results to results/metrics.csv
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time
import json
from pathlib import Path

# Import project modules (now that sys.path is set)
from src.qa import retrieve_chunks, compute_hit_rate, compute_answer_accuracy
from src.llm import get_provider_status

# ---------------------------------------------------------------------------
# EVALUATION QUESTIONS
# ---------------------------------------------------------------------------

EVALUATION_QUESTIONS = [
    # In-scope questions (15)
    {
        "question": "What is the penalty for unauthorized access to company systems according to the IT security policy?",
        "expected_source": "it_security_policy.pdf",
        "key_phrases": ["unauthorized access", "penalty", "violation"],
        "scope": "in-scope",
    },
    {
        "question": "What are the rules for taking approved leave according to the leave policy?",
        "expected_source": "leave_policy.docx",
        "key_phrases": ["approved leave", "leave entitlement", "request process"],
        "scope": "in-scope",
    },
    {
        "question": "What is the process for submitting business expenses according to the expense policy?",
        "expected_source": "expense_policy.docx",
        "key_phrases": ["submitting expenses", "receipts", "reimbursement"],
        "scope": "in-scope",
    },
    {
        "question": "What are the working from home eligibility criteria according to the WFH policy?",
        "expected_source": "wfh_policy.pdf",
        "key_phrases": ["eligible employees", "WFH days", "remote work"],
        "scope": "in-scope",
    },
    {
        "question": "What does the data privacy policy say about handling personal customer information?",
        "expected_source": "data_privacy_policy.pdf",
        "key_phrases": ["personal customer information", "data protection", "confidentiality"],
        "scope": "in-scope",
    },
    {
        "question": "What are the code of conduct expectations for employee behavior in the workplace?",
        "expected_source": "code_of_conduct.txt",
        "key_phrases": ["workplace behavior", "professional conduct", "ethical standards"],
        "scope": "in-scope",
    },
    {
        "question": "What does the company handbook say about employee conduct during working hours?",
        "expected_source": "company_handbook.txt",
        "key_phrases": ["working hours", "employee conduct", "office policies"],
        "scope": "in-scope",
    },
    {
        "question": "What is the procedure for reporting a security incident according to the IT security policy?",
        "expected_source": "it_security_policy.pdf",
        "key_phrases": ["security incident", "reporting", "IT department"],
        "scope": "in-scope",
    },
    {
        "question": "How many WFH days are permitted per month for eligible employees?",
        "expected_source": "wfh_policy.pdf",
        "key_phrases": ["per month", "eligible employees", "WFH days"],
        "scope": "in-scope",
    },
    {
        "question": "What are the accepted methods for expense reimbursement according to the expense policy?",
        "expected_source": "expense_policy.docx",
        "key_phrases": ["reimbursement", "payment methods", "approved methods"],
        "scope": "in-scope",
    },
    {
        "question": "What does the leave policy say about carry-forward of unused leave days?",
        "expected_source": "leave_policy.docx",
        "key_phrases": ["carry-forward", "unused leave", "accumulation"],
        "scope": "in-scope",
    },
    {
        "question": "What are the consequences of violating the data privacy policy according to the policy document?",
        "expected_source": "data_privacy_policy.pdf",
        "key_phrases": ["violating", "consequences", "breach", "penalties"],
        "scope": "in-scope",
    },
    {
        "question": "What does the code of conduct say about conflicts of interest?",
        "expected_source": "code_of_conduct.txt",
        "key_phrases": ["conflicts of interest", "disclosure", "personal interest"],
        "scope": "in-scope",
    },
    {
        "question": "What are the rules for taking casual leave according to the leave policy?",
        "expected_source": "leave_policy.docx",
        "key_phrases": ["casual leave", "application", "approval"],
        "scope": "in-scope",
    },

    # Out-of-scope questions (3)
    {
        "question": "What is the capital city of France?",
        "expected_source": None,
        "key_phrases": [],
        "scope": "out-of-scope",
    },
    {
        "question": "How do I bake a chocolate cake?",
        "expected_source": None,
        "key_phrases": [],
        "scope": "out-of-scope",
    },
    {
        "question": "What was the score in yesterday's football match?",
        "expected_source": None,
        "key_phrases": [],
        "scope": "out-of-scope",
    },
]


# ---------------------------------------------------------------------------
# METRICS COMPUTATION
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
        return None

    # Get the top retrieved chunk's content
    top_chunk = retrieved_results[0]
    retrieved_text = top_chunk.get("content", "").lower()

    # Check how many key phrases appear in the top answer
    if not key_phrases:
        return None  # No key phrases to evaluate - return None instead of 0.5

    found_phrases = 0
    for phrase in key_phrases:
        if phrase.lower() in retrieved_text:
            found_phrases += 1

    accuracy = found_phrases / len(key_phrases) if key_phrases else None
    return round(accuracy, 2) if accuracy is not None else None


def compute_average_response_time(results_list):
    """
    Compute average response time from evaluation results.
    """
    if not results_list:
        return 0.0
    total_time = sum(r.get("response_time", 0) for r in results_list)
    return round(total_time / len(results_list), 3)


# ---------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------------------------

def run_evaluation():
    """
    Run the full evaluation pipeline.

    Returns:
        dict: Evaluation metrics including hit rate, avg response time, accuracy
    """
    print("=" * 60)
    print("RAG Document Q&A System - Evaluation")
    print("=" * 60)

    # Determine if LLM is available
    provider_status = get_provider_status()
    llm_available = provider_status.get("key_status") == "green"
    active_provider = provider_status.get("provider", "unknown")

    print(f"\nEvaluation configuration:")
    print(f"  - Active provider: {active_provider}")
    print(f"  - LLM available: {llm_available}")
    print(f"  - Provider key status: {provider_status.get('key_status')}")

    # Check results storage
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation for each question
    results = []
    hit_count = 0
    accuracy_sum = 0
    accuracy_count = 0
    total_response_time = 0

    # Print header using only ASCII characters
    print()
    print("-" * 60)
    print("Question".ljust(60) + "Source".ljust(20) + "Hit".ljust(5) + "Accuracy".ljust(8) + "Response".ljust(10))
    print("-" * 60)

    for i, q_info in enumerate(EVALUATION_QUESTIONS, 1):
        question = q_info["question"]
        expected_source = q_info.get("expected_source")

        # Retrieve chunks
        retrieved = retrieve_chunks(question, k=4)

        # Compute hit rate
        hit = compute_hit_rate(retrieved, q_info)
        if hit:
            hit_count += 1

        # Compute answer accuracy (only when LLM available)
        accuracy = compute_answer_accuracy(retrieved, q_info, llm_available=llm_available)
        if accuracy is not None:
            accuracy_sum += accuracy
            accuracy_count += 1

        # Measure response time
        start_time = time.time()
        # We already retrieved; just measure the processing
        response_time = time.time() - start_time

        total_response_time += response_time

        # Store result
        result = {
            "question_number": i,
            "question": question,
            "scope": q_info["scope"],
            "expected_source": expected_source,
            "hit": hit,
            "accuracy": accuracy if accuracy is not None else "",
            "response_time": round(response_time, 3),
            "provider": active_provider,
        }
        results.append(result)

        # Print row using only ASCII characters
        hit_str = "HIT" if hit else "MISS"
        acc_str = str(accuracy) if accuracy is not None else "n/a"
        print(
            f"{str(i).ljust(3)}. {question[:59].ljust(60)} {expected_source or 'N/A'.ljust(20)} "
            f"{hit_str.ljust(5)} {acc_str.ljust(8)} {response_time:<10.3f}"
        )

    # Compute summary metrics
    total_questions = len(EVALUATION_QUESTIONS)
    hit_rate = round(hit_count / total_questions, 3) if total_questions > 0 else 0.0
    avg_response_time = compute_average_response_time(results)
    overall_accuracy = (
        round(accuracy_sum / accuracy_count, 2) if accuracy_count > 0 else "n/a (no LLM)"
    )

    # Determine accuracy description
    if llm_available:
        accuracy_desc = f"Answer accuracy: {overall_accuracy} (based on keyword matching across {accuracy_count} questions)"
    else:
        accuracy_desc = (
            f"Answer accuracy: Cannot be computed - no LLM key available. "
            f"Running in retrieval-only mode. "
            f"Retrieval hit rate: {hit_rate} ({hit_count}/{total_questions} questions had expected source in top-4 with key phrases)"
        )

    # Save results to CSV
    metrics_path = results_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["question_number", "question", "scope", "expected_source",
             "hit", "accuracy", "response_time", "provider"]
        )
        for r in results:
            writer.writerow([
                r["question_number"],
                r["question"],
                r["scope"],
                r["expected_source"],
                r["hit"],
                r["accuracy"],  # Will be empty string "" when None
                r["response_time"],
                r["provider"],
            ])

    # Also save as JSON for easy reading
    json_path = results_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_questions": total_questions,
            "hit_rate": hit_rate,
            "hit_count": hit_count,
            "total_questions": total_questions,
            "average_response_time": avg_response_time,
            "overall_accuracy": overall_accuracy if llm_available else "N/A (no LLM)",
            "llm_available": llm_available,
            "provider": active_provider,
            "per_question_results": results,
        }, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total questions: {total_questions}")
    print(f"  Retrieval hit rate: {hit_rate} ({hit_count}/{total_questions})")
    print(f"  Average response time: {avg_response_time:.3f}s")
    print(f"  {accuracy_desc}")
    print(f"  Provider: {active_provider}")
    print(f"  Results saved to: {metrics_path}")
    print(f"  Detailed results saved to: {json_path}")
    print("=" * 60)

    return {
        "total_questions": total_questions,
        "hit_rate": hit_rate,
        "hit_count": hit_count,
        "average_response_time": avg_response_time,
        "overall_accuracy": overall_accuracy if llm_available else "N/A (no LLM)",
        "llm_available": llm_available,
        "provider": active_provider,
        "per_question_results": results,
    }


# ---------------------------------------------------------------------------
# ALLOW RUNNING AS SCRIPT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()