"""
intent_detector.py
Classifies user intent and decides whether RAG retrieval is needed.

Intent categories:
  - career:     questions about work, job, profession, business
  - love:       questions about relationships, romance, partner
  - spiritual:  questions about spirituality, meditation, dharma
  - planetary:  questions about specific planets affecting the user
  - zodiac:     questions about personality traits, zodiac characteristics
  - stress:     questions about stress, anxiety, emotions today
  - summary:    meta-questions asking for a conversation recap
  - general:    generic greeting, chitchat — NO retrieval needed

RAG is skipped for:
  - Purely meta/conversational turns (summary, greetings, follow-ups)
  - Repeated context that was already retrieved in the same session
"""

import re
from typing import Tuple

# Keyword patterns for intent matching
INTENT_PATTERNS = {
    "career": [
        r"\bcareer\b", r"\bjob\b", r"\bwork\b", r"\bprofession\b",
        r"\bbusiness\b", r"\bpromotion\b", r"\boffice\b", r"\bmonth\b.*\bcareer\b",
        r"\bcareer\b.*\bmonth\b", r"\bnaukri\b", r"\bkaam\b"
    ],
    "love": [
        r"\blove\b", r"\brelationship\b", r"\bpartner\b", r"\bromance\b",
        r"\bmarriage\b", r"\bwife\b", r"\bhusband\b", r"\bboyfriend\b",
        r"\bgirlfriend\b", r"\bpyar\b", r"\brishta\b", r"\bshadi\b"
    ],
    "spiritual": [
        r"\bspiritual\b", r"\bmeditation\b", r"\bdharma\b", r"\bkarma\b",
        r"\bmoksha\b", r"\binner\b", r"\bsoul\b", r"\bpeace\b",
        r"\badhyatm\b", r"\bdhyan\b"
    ],
    "planetary": [
        r"\bplanet\b", r"\bsaturn\b", r"\bjupiter\b", r"\bmars\b",
        r"\bvenus\b", r"\bmercury\b", r"\bsun\b", r"\bmoon\b",
        r"\brahu\b", r"\bketu\b", r"\bshani\b", r"\bmangal\b"
    ],
    "zodiac": [
        r"\bzodiac\b", r"\bsign\b", r"\btrait\b", r"\bpersonality\b",
        r"\bleo\b", r"\baries\b", r"\btaurus\b", r"\bgemini\b",
        r"\bcancer\b", r"\bvirgo\b", r"\blibra\b", r"\bscorpio\b",
        r"\bsagittarius\b", r"\bcapricorn\b", r"\baquarius\b", r"\bpisces\b",
        r"\brashi\b"
    ],
    "stress": [
        r"\bstress\b", r"\banxiety\b", r"\btense\b", r"\bworried\b",
        r"\bdifficult\b", r"\bhard\b.*\bday\b", r"\bbad\b.*\bday\b",
        r"\btakleef\b", r"\bpareshan\b", r"\bchinta\b"
    ],
    "summary": [
        r"\bsummar\w*\b", r"\bwhat.*told\b", r"\brecap\b",
        r"\brepeat\b", r"\bsaying again\b", r"\bdobara\b",
        r"\bpehle\b.*\bbola\b"
    ],
    "greeting": [
        r"^\s*(hi|hello|hey|namaste|namaskar|hola)\b",
        r"^\s*how are you\b",
        r"^\s*good\s*(morning|evening|afternoon|night)\b"
    ]
}

# Intent to knowledge source mapping
INTENT_TO_SOURCES = {
    "career":    ["career_guidance", "zodiac_traits", "planetary_impacts"],
    "love":      ["love_guidance", "zodiac_traits", "planetary_impacts"],
    "spiritual": ["spiritual_guidance", "zodiac_traits"],
    "planetary": ["planetary_impacts", "zodiac_traits"],
    "zodiac":    ["zodiac_traits"],
    "stress":    ["career_guidance", "love_guidance", "spiritual_guidance", "planetary_impacts"],
    "summary":   [],   # No retrieval — use conversation memory
    "greeting":  [],   # No retrieval — purely conversational
    "general":   ["zodiac_traits"],
}

# Intents that should NOT trigger RAG retrieval
NO_RETRIEVAL_INTENTS = {"summary", "greeting"}


def detect_intent(message: str) -> str:
    """
    Detect the user's intent from their message.
    Returns one of the intent keys.
    """
    msg_lower = message.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, msg_lower):
                return intent

    return "general"


def should_retrieve(intent: str, session_retrieved_intents: list) -> bool:
    """
    Decide whether RAG retrieval is needed for this turn.

    Rules:
    - Never retrieve for 'summary' or 'greeting' intents
    - Avoid duplicate retrieval for the same intent in short windows (last 2 turns)
      This avoids redundant retrieval and unnecessary token usage
    - Otherwise retrieve

    Returns True if retrieval should happen.
    """
    if intent in NO_RETRIEVAL_INTENTS:
        return False

    # Avoid re-retrieving same intent in very recent turns (last 2) — saves tokens
    recent = session_retrieved_intents[-2:] if len(session_retrieved_intents) >= 2 else session_retrieved_intents
    if intent in recent and intent not in ["planetary", "stress"]:
        return False

    return True


def get_retrieval_sources(intent: str) -> list:
    """Return the list of knowledge source names relevant to this intent."""
    return INTENT_TO_SOURCES.get(intent, ["zodiac_traits"])


def classify_message(message: str, session_retrieved_intents: list) -> Tuple[str, bool, list]:
    """
    Full classification pipeline.

    Returns:
        intent (str): detected intent
        do_retrieve (bool): whether to perform RAG retrieval
        sources (list): knowledge sources to query
    """
    intent = detect_intent(message)
    do_retrieve = should_retrieve(intent, session_retrieved_intents)
    sources = get_retrieval_sources(intent) if do_retrieve else []
    return intent, do_retrieve, sources
