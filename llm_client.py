"""
llm_client.py
LLM abstraction layer.

Supports:
  - OpenAI GPT-4o / GPT-3.5-turbo
  - Stub mode (rule-based responses) when no API key is present
  - Hindi language response toggle
  - Retry logic with exponential backoff
  - Safe error fallbacks
"""

import os
import time
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Try importing openai
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai library not installed. Running in stub mode.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_RETRIES = 3
RETRY_DELAY = 1.5   # seconds


# ---------------------------------------------------------------------------
# Hindi system prompt instruction
# ---------------------------------------------------------------------------

HINDI_INSTRUCTION = (
    "Please respond entirely in Hindi (Devanagari script). "
    "Use respectful and warm language."
)

SYSTEM_PROMPT_TEMPLATE = """You are Astro, a warm and wise AI astrologer specializing in Vedic astrology.
You provide personalized astrological guidance based on the user's birth chart, zodiac sign, and current planetary influences.

Your style:
- Warm, empathetic, and encouraging
- Specific to the user's zodiac sign and profile
- Grounded in the retrieved astrological context (if provided)
- Never make absolute predictions — speak in terms of tendencies and possibilities

User Profile:
{user_profile_summary}

{conversation_context}

{hindi_instruction}

When retrieved astrological context is provided, use it to ground your response.
When no context is provided (conversational turns), rely on your astrological knowledge and the conversation history.
Always personalize your response to the user's zodiac sign ({zodiac_sign}).
"""


class LLMClient:
    """
    Abstraction over OpenAI API.
    Falls back to rule-based stub responses when API key is absent.
    """

    def __init__(self):
        self._client = None
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            self._client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info(f"OpenAI LLM client initialized with model: {DEFAULT_MODEL}")
        else:
            logger.info("LLM running in STUB mode (no OpenAI key detected).")

    def generate(
        self,
        user_message: str,
        system_prompt: str,
        chat_history: List[Dict],
        max_tokens: int = 512,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            user_message: The latest user message
            system_prompt: Constructed system prompt with context
            chat_history: List of prior turns as OpenAI-format messages
            max_tokens: Token budget for response

        Returns:
            Response string
        """
        if self._client:
            return self._openai_generate(user_message, system_prompt, chat_history, max_tokens)
        else:
            return self._stub_generate(user_message, system_prompt)

    def _openai_generate(self, user_message, system_prompt, chat_history, max_tokens) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"OpenAI attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    logger.error("All OpenAI retries exhausted.")
                    return self._error_fallback(user_message)

    def _stub_generate(self, user_message: str, system_prompt: str) -> str:
        """
        Rule-based stub response when OpenAI is unavailable.
        Extracts context hints from the system prompt to produce semi-relevant responses.
        """
        msg_lower = user_message.lower()
        is_hindi = "Hindi" in system_prompt

        # Extract zodiac from system prompt
        zodiac = "your zodiac sign"
        for line in system_prompt.splitlines():
            if "Zodiac Sign:" in line:
                zodiac = line.split("Zodiac Sign:")[-1].split("(")[0].strip()
                break

        if any(w in msg_lower for w in ["career", "job", "work", "kaam", "naukri"]):
            if is_hindi:
                return (f"आपके लिए यह समय करियर में मेहनत और एकाग्रता का है। "
                        f"{zodiac} राशि के लिए गुरु ग्रह का प्रभाव अवसर ला रहा है। "
                        f"नई जिम्मेदारियाँ आ सकती हैं — उन्हें आत्मविश्वास से स्वीकार करें।")
            return (f"As a {zodiac}, this is a powerful time for career growth. "
                    f"Jupiter's influence opens doors to new opportunities and recognition. "
                    f"Focus on teamwork and clear communication — success is within reach.")

        elif any(w in msg_lower for w in ["love", "relationship", "partner", "pyar", "rishta"]):
            if is_hindi:
                return (f"आपके प्रेम जीवन में शुक्र ग्रह सकारात्मक ऊर्जा ला रहा है। "
                        f"{zodiac} राशि के लिए भावनात्मक संवाद और समझ बहुत महत्वपूर्ण है। "
                        f"खुलकर अपनी भावनाएं व्यक्त करें।")
            return (f"For {zodiac}, Venus is currently enhancing your emotional connections. "
                    f"Open and honest communication will strengthen your relationship. "
                    f"This is a beautiful time for deepening bonds.")

        elif any(w in msg_lower for w in ["stress", "anxious", "worried", "pareshan", "chinta"]):
            if is_hindi:
                return (f"आज का तनाव शनि और मंगल के प्रभाव से हो सकता है। "
                        f"ध्यान और गहरी सांस लेने का अभ्यास करें। "
                        f"{zodiac} राशि के लिए आज का दिन संयम और धैर्य का है।")
            return (f"The stress you're feeling today may be influenced by Saturn's current transit. "
                    f"As a {zodiac}, grounding practices like meditation can help significantly. "
                    f"Trust that this challenging phase will pass and bring growth.")

        elif any(w in msg_lower for w in ["spiritual", "meditation", "karma", "dhyan"]):
            if is_hindi:
                return (f"आपकी आध्यात्मिक यात्रा में केतु ग्रह मार्गदर्शन दे रहा है। "
                        f"ध्यान और कृतज्ञता से आंतरिक शांति मिलेगी। "
                        f"{zodiac} राशि के लिए यह आत्मचिंतन का उत्कृष्ट समय है।")
            return (f"Your spiritual path is deeply influenced by Ketu's energy right now. "
                    f"As a {zodiac}, this is an excellent time for inner reflection and meditation. "
                    f"The universe is guiding you toward greater wisdom.")

        elif any(w in msg_lower for w in ["summar", "told", "recap", "dobara"]):
            if is_hindi:
                return "अभी तक हमने करियर, प्रेम जीवन और आपकी राशि के बारे में बात की है। क्या आप किसी विशेष विषय पर और जानकारी चाहते हैं?"
            return (f"So far in our conversation, we've discussed your astrological profile as a {zodiac}. "
                    f"I've shared insights about planetary influences affecting your life. "
                    f"Is there a specific area you'd like to explore further?")

        else:
            if is_hindi:
                return (f"नमस्ते! मैं Astro हूं, आपका व्यक्तिगत ज्योतिष सहायक। "
                        f"{zodiac} राशि के रूप में, आपके जीवन में ग्रहों का विशेष प्रभाव है। "
                        f"आप करियर, प्रेम, या आध्यात्मिकता के बारे में पूछ सकते हैं।")
            return (f"Namaste! As a {zodiac}, the stars have a fascinating story to tell about your journey. "
                    f"I'm here to guide you through astrological insights about career, love, spirituality, "
                    f"and planetary influences. What would you like to explore today?")

    def _error_fallback(self, user_message: str) -> str:
        return (
            "I apologize — I'm experiencing a temporary connection issue. "
            "Please try again in a moment. The stars are still watching over you! 🌟"
        )


def build_system_prompt(
    user_profile_summary: str,
    zodiac_sign: str,
    conversation_context: str,
    retrieved_context: str,
    preferred_language: str = "en",
) -> str:
    """Construct the full system prompt for the LLM."""
    hindi_instruction = HINDI_INSTRUCTION if preferred_language == "hi" else ""

    if retrieved_context:
        rag_section = f"\nRetrieved Astrological Knowledge:\n{retrieved_context}\n"
    else:
        rag_section = ""

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        user_profile_summary=user_profile_summary,
        conversation_context=conversation_context,
        hindi_instruction=hindi_instruction,
        zodiac_sign=zodiac_sign,
    )
    return prompt + rag_section


# Singleton
_llm_client: LLMClient = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
