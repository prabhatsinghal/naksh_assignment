"""
conversation_manager.py
Manages per-session multi-turn conversation memory.

Features:
  - Stores full turn history per session_id
  - Windowed memory: keeps last N turns in active context
  - Controlled memory growth via summarization stubs
  - Tracks which intents have been retrieved in this session
  - Thread-safe in-process store (replace with Redis/DB for production)
"""

import threading
from datetime import datetime
from typing import List, Dict, Optional

# Max number of turns kept in full detail
WINDOW_SIZE = 10

# After this many turns, a summary is injected to compress older context
SUMMARIZE_AFTER = 20


class Turn:
    """Represents a single conversational turn (user + assistant)."""

    def __init__(self, user_msg: str, assistant_msg: str, intent: str, retrieval_used: bool):
        self.user_msg = user_msg
        self.assistant_msg = assistant_msg
        self.intent = intent
        self.retrieval_used = retrieval_used
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "user": self.user_msg,
            "assistant": self.assistant_msg,
            "intent": self.intent,
            "retrieval_used": self.retrieval_used,
            "timestamp": self.timestamp,
        }


class Session:
    """Holds the full state for a single conversation session."""

    def __init__(self, session_id: str, user_profile: dict):
        self.session_id = session_id
        self.user_profile = user_profile
        self.turns: List[Turn] = []
        self.summary: Optional[str] = None          # Compressed older context
        self.retrieved_intents: List[str] = []       # Track retrieval history
        self.created_at = datetime.utcnow().isoformat()

    def add_turn(self, user_msg: str, assistant_msg: str, intent: str, retrieval_used: bool):
        turn = Turn(user_msg, assistant_msg, intent, retrieval_used)
        self.turns.append(turn)
        if retrieval_used:
            self.retrieved_intents.append(intent)
        self._maybe_summarize()

    def _maybe_summarize(self):
        """Compress older turns into a summary to control memory growth."""
        if len(self.turns) > SUMMARIZE_AFTER:
            old_turns = self.turns[:-WINDOW_SIZE]
            summary_lines = [self.summary] if self.summary else []
            for t in old_turns:
                summary_lines.append(f"User asked about {t.intent}: '{t.user_msg[:80]}...'")
            self.summary = "Earlier conversation summary: " + " | ".join(summary_lines[-10:])
            self.turns = self.turns[-WINDOW_SIZE:]

    def get_window(self, n: int = WINDOW_SIZE) -> List[Turn]:
        """Return the last N turns for active context."""
        return self.turns[-n:]

    def get_chat_history_for_llm(self, n: int = WINDOW_SIZE) -> List[Dict]:
        """Format recent turns as OpenAI-style messages list."""
        messages = []
        for turn in self.get_window(n):
            messages.append({"role": "user", "content": turn.user_msg})
            messages.append({"role": "assistant", "content": turn.assistant_msg})
        return messages

    def get_context_summary(self) -> str:
        """Return a brief text summary of what's been discussed so far."""
        if not self.turns:
            return "This is the start of the conversation."
        topics = list(dict.fromkeys([t.intent for t in self.turns]))
        lines = [f"Conversation topics so far: {', '.join(topics)}."]
        if self.summary:
            lines.insert(0, self.summary)
        recent = self.turns[-3:]
        for t in recent:
            lines.append(f"Recently, user asked: '{t.user_msg[:100]}'")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_profile": self.user_profile,
            "turns": [t.to_dict() for t in self.turns],
            "summary": self.summary,
            "retrieved_intents": self.retrieved_intents,
            "created_at": self.created_at,
        }


class ConversationManager:
    """
    In-memory multi-session conversation store.
    Thread-safe via a per-session lock.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create_session(self, session_id: str, user_profile: dict) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id, user_profile)
            else:
                # Update profile if new info arrives (e.g., language preference change)
                existing = self._sessions[session_id]
                for key, val in user_profile.items():
                    if val:
                        existing.user_profile[key] = val
            return self._sessions[session_id]

    def record_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        intent: str,
        retrieval_used: bool
    ):
        session = self._sessions.get(session_id)
        if session:
            session.add_turn(user_msg, assistant_msg, intent, retrieval_used)

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# Singleton
_manager: ConversationManager = None


def get_conversation_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager
