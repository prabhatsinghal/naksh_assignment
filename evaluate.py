"""
evaluate.py
Basic evaluation script for the Astro RAG Agent.

Demonstrates:
  Case A — Retrieval HELPED  : Domain-specific question about career for a Leo
  Case B — Retrieval HURT    : Meta-question asking for a conversation recap
  Case C — Correct bypass    : Greeting (no retrieval needed)

Run: python evaluate.py
(Requires the FastAPI server to be running OR uses internal module imports directly)
"""

import sys
import json
import logging

logging.basicConfig(level=logging.WARNING)

# Import modules directly (no server needed for evaluation)
from profile_builder import build_user_profile, profile_summary
from intent_detector import classify_message
from rag_engine import get_rag_engine
from llm_client import get_llm_client, build_system_prompt
from conversation_manager import get_conversation_manager


def run_case(
    label: str,
    message: str,
    birth_date: str = "1995-08-20",
    name: str = "Ritika",
    birth_place: str = "Jaipur, India",
    language: str = "en",
    session_id: str = "eval_session",
    session_retrieved_history: list = None,
):
    print(f"\n{'='*60}")
    print(f"CASE: {label}")
    print(f"Message: \"{message}\"")
    print("-" * 60)

    if session_retrieved_history is None:
        session_retrieved_history = []

    # Build profile
    raw_profile = {
        "name": name,
        "birth_date": birth_date,
        "birth_place": birth_place,
        "preferred_language": language,
    }
    profile = build_user_profile(raw_profile)
    zodiac = profile["zodiac_sign"]
    print(f"Zodiac: {zodiac}")

    # Intent detection
    intent, do_retrieve, sources = classify_message(message, session_retrieved_history)
    print(f"Intent: {intent}")
    print(f"Should Retrieve: {do_retrieve}")
    print(f"Sources Targeted: {sources}")

    # Retrieval
    retrieved_context_str = ""
    context_used = []
    avg_score = 0.0

    if do_retrieve and sources:
        engine = get_rag_engine()
        augmented_query = f"{message} {zodiac} zodiac"
        chunks, avg_score = engine.retrieve(
            query=augmented_query,
            sources=sources,
            top_k=3
        )
        if chunks:
            retrieved_context_str = engine.format_context(chunks)
            context_used = list(dict.fromkeys([c["source"] for c in chunks]))
            print(f"Chunks Retrieved: {len(chunks)} (avg_score={avg_score:.3f})")
            print(f"Sources Used: {context_used}")
        else:
            print("No chunks above threshold found.")
            do_retrieve = False
    else:
        print("Retrieval SKIPPED (intent-aware decision)")

    # Generate response
    profile_sum = profile_summary(profile)
    system_prompt = build_system_prompt(
        user_profile_summary=profile_sum,
        zodiac_sign=zodiac,
        conversation_context="Start of evaluation session.",
        retrieved_context=retrieved_context_str,
        preferred_language=language,
    )
    llm = get_llm_client()
    response = llm.generate(
        user_message=message,
        system_prompt=system_prompt,
        chat_history=[],
        max_tokens=300,
    )

    print(f"\nResponse:\n{response}")

    # Analysis
    print("\nEVALUATION:")
    if do_retrieve:
        print(f"  ✅ Retrieval was used (avg similarity: {avg_score:.3f})")
        if avg_score >= 0.45:
            print("  ✅ High relevance — retrieval HELPED the response quality")
        elif avg_score >= 0.30:
            print("  ⚠️  Moderate relevance — retrieval was acceptable")
        else:
            print("  ❌ Low relevance — retrieval may have HURT (noise introduced)")
    else:
        print("  ✅ Retrieval correctly BYPASSED — response based on memory/conversation")

    return {
        "case": label,
        "intent": intent,
        "retrieval_used": do_retrieve,
        "context_used": context_used,
        "avg_similarity": avg_score,
        "response_preview": response[:120],
    }


def main():
    print("🔭 Astro Agent — RAG Evaluation Suite")
    print("=" * 60)

    results = []

    # -----------------------------------------------------------------------
    # Case A: Retrieval HELPED
    # Domain-specific query about career for Leo — RAG adds real value here
    # -----------------------------------------------------------------------
    r1 = run_case(
        label="A — Retrieval HELPED (career query for Leo)",
        message="What should I focus on in my career this month?",
        birth_date="1995-08-20",   # Leo
        session_retrieved_history=[],
    )
    results.append(r1)

    # -----------------------------------------------------------------------
    # Case B: Retrieval HURT / Should be Bypassed
    # Meta-question asking for summary — RAG adds no value, just wastes tokens
    # -----------------------------------------------------------------------
    r2 = run_case(
        label="B — Retrieval BYPASSED (summary meta-query)",
        message="Summarize what you've told me so far",
        birth_date="1990-03-15",   # Pisces
        session_retrieved_history=["career", "love"],
    )
    results.append(r2)

    # -----------------------------------------------------------------------
    # Case C: Greeting — purely conversational, no retrieval
    # -----------------------------------------------------------------------
    r3 = run_case(
        label="C — No Retrieval (greeting/chit-chat)",
        message="Hello! Good morning.",
        birth_date="1998-11-10",   # Scorpio
        session_retrieved_history=[],
    )
    results.append(r3)

    # -----------------------------------------------------------------------
    # Case D: Hindi language toggle
    # -----------------------------------------------------------------------
    r4 = run_case(
        label="D — Hindi Language Toggle (love query in Hindi)",
        message="कौन सा ग्रह मेरे प्रेम जीवन को प्रभावित कर रहा है?",
        birth_date="1993-04-05",   # Aries
        language="hi",
        session_retrieved_history=[],
    )
    results.append(r4)

    # -----------------------------------------------------------------------
    # Case E: Repeated intent — retrieval correctly avoided (token cost)
    # -----------------------------------------------------------------------
    r5 = run_case(
        label="E — Retrieval SKIPPED (same intent repeated recently)",
        message="Can you tell me more about my career?",
        birth_date="1990-06-15",   # Gemini
        session_retrieved_history=["career", "career"],   # Already retrieved twice
    )
    results.append(r5)

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for r in results:
        retrieval_label = "✅ Used" if r["retrieval_used"] else "⛔ Bypassed"
        score_str = f"(sim={r['avg_similarity']:.3f})" if r["retrieval_used"] else ""
        print(f"  [{r['case'][:40]:<40}] Retrieval: {retrieval_label} {score_str}")


if __name__ == "__main__":
    main()
