"""
app.py
FastAPI entry point for the Astro Conversational Insight Agent.

Endpoints:
  POST /chat          — Main conversational endpoint
  GET  /session/{id}  — Inspect session state
  DELETE /session/{id}— Clear session memory
  GET  /health        — Health check
"""

import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from profile_builder import build_user_profile, profile_summary
from intent_detector import classify_message
from rag_engine import get_rag_engine
from conversation_manager import get_conversation_manager
from llm_client import get_llm_client, build_system_prompt

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Astro Conversational Insight Agent",
    description="Multi-turn RAG-powered Vedic astrology chatbot with personalization.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    name: str = Field(default="User", description="User's name")
    birth_date: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    birth_time: Optional[str] = Field(default="", description="Time of birth HH:MM")
    birth_place: Optional[str] = Field(default="", description="City, Country of birth")
    preferred_language: Optional[str] = Field(default="en", description="'en' or 'hi'")
    moon_sign: Optional[str] = Field(default=None)
    goals: Optional[List[str]] = Field(default=[])


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User's message")
    user_profile: UserProfile


class ChatResponse(BaseModel):
    response: str
    zodiac: str
    context_used: List[str]
    retrieval_used: bool
    intent: str
    session_id: str


# ---------------------------------------------------------------------------
# Startup: Preload RAG engine
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Pre-loading RAG knowledge base on startup...")
    engine = get_rag_engine()
    logger.info(f"RAG engine loaded with {len(engine.chunks)} chunks.")


# ---------------------------------------------------------------------------
# Error Handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."}
    )


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, summary="Send a message to Astro")
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.

    Steps:
    1. Build/update user profile (zodiac, age, etc.)
    2. Detect intent from message
    3. Decide if RAG retrieval is needed
    4. If yes, retrieve relevant astrological context
    5. Build full prompt (profile + memory + context)
    6. Generate LLM response
    7. Store turn in session memory
    8. Return structured response
    """
    # --- Validate ---
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not request.user_profile.birth_date:
        raise HTTPException(status_code=400, detail="birth_date is required.")

    # --- 1. Build profile ---
    raw_profile = request.user_profile.model_dump()
    profile = build_user_profile(raw_profile)
    zodiac = profile["zodiac_sign"]
    language = profile.get("preferred_language", "en")

    # --- 2. Session management ---
    session_manager = get_conversation_manager()
    session = session_manager.get_or_create_session(request.session_id, profile)

    # --- 3. Intent detection ---
    intent, do_retrieve, sources = classify_message(
        request.message,
        session.retrieved_intents
    )
    logger.info(f"[{request.session_id}] Intent: {intent} | Retrieve: {do_retrieve} | Sources: {sources}")

    # --- 4. RAG Retrieval ---
    retrieved_chunks = []
    retrieved_context_str = ""
    avg_score = 0.0
    context_used = []

    if do_retrieve and sources:
        engine = get_rag_engine()
        # Augment query with zodiac for personalization
        augmented_query = f"{request.message} {zodiac} zodiac"
        retrieved_chunks, avg_score = engine.retrieve(
            query=augmented_query,
            sources=sources,
            top_k=4
        )

        if retrieved_chunks:
            retrieved_context_str = engine.format_context(retrieved_chunks)
            context_used = list(dict.fromkeys([c["source"] for c in retrieved_chunks]))
            logger.info(
                f"[{request.session_id}] Retrieved {len(retrieved_chunks)} chunks "
                f"(avg_score={avg_score:.3f}) from: {context_used}"
            )
        else:
            logger.info(f"[{request.session_id}] No chunks above threshold — skipping retrieval context.")
            do_retrieve = False   # Downgrade: retrieval found nothing useful

    # --- 5. Build system prompt ---
    profile_sum = profile_summary(profile)
    convo_context = session.get_context_summary()
    chat_history = session.get_chat_history_for_llm(n=6)

    system_prompt = build_system_prompt(
        user_profile_summary=profile_sum,
        zodiac_sign=zodiac,
        conversation_context=convo_context,
        retrieved_context=retrieved_context_str,
        preferred_language=language,
    )

    # --- 6. LLM generation ---
    llm = get_llm_client()
    response_text = llm.generate(
        user_message=request.message,
        system_prompt=system_prompt,
        chat_history=chat_history,
        max_tokens=600 if language == "hi" else 500,
    )

    # --- 7. Record turn ---
    session_manager.record_turn(
        session_id=request.session_id,
        user_msg=request.message,
        assistant_msg=response_text,
        intent=intent,
        retrieval_used=do_retrieve,
    )

    # --- 8. Return response ---
    return ChatResponse(
        response=response_text,
        zodiac=zodiac,
        context_used=context_used,
        retrieval_used=do_retrieve,
        intent=intent,
        session_id=request.session_id,
    )


# ---------------------------------------------------------------------------
# GET /session/{session_id}
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}", summary="Inspect session state")
async def get_session(session_id: str):
    """Return the full session state including turn history."""
    manager = get_conversation_manager()
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.to_dict()


# ---------------------------------------------------------------------------
# DELETE /session/{session_id}
# ---------------------------------------------------------------------------

@app.delete("/session/{session_id}", summary="Clear a session")
async def delete_session(session_id: str):
    """Delete a session and its memory."""
    manager = get_conversation_manager()
    if not manager.get_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    manager.delete_session(session_id)
    return {"message": f"Session '{session_id}' deleted successfully."}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health():
    engine = get_rag_engine()
    return {
        "status": "ok",
        "rag_chunks_loaded": len(engine.chunks),
        "sessions_active": len(get_conversation_manager().list_sessions()),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
