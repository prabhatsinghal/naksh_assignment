# 🔭 Astro Conversational Insight Agent

A multi-turn conversational AI service powered by **Retrieval-Augmented Generation (RAG)** and **Vedic Astrology** knowledge.

---

## ✨ Features

| Feature | Implementation |
|---|---|
| **Multi-turn Memory** | Session-based windowed memory (last 10 turns) with auto-summarization |
| **Intent-Aware RAG** | Retrieval triggered only when it adds value (not always-on) |
| **Personalization** | Zodiac sign, moon sign, age, goals all used in prompt |
| **Semantic Retrieval** | FAISS + `sentence-transformers` with cosine similarity |
| **Hindi Language** | Full Hindi toggle via `preferred_language: "hi"` |
| **Stub Mode** | Works without an OpenAI key using rule-based fallback |
| **Evaluation Suite** | Built-in eval showing when retrieval helped vs hurt |

---

## 📁 Project Structure

```
astro_agent/
├── app.py                    # FastAPI application (entry point)
├── profile_builder.py        # Zodiac calculation & user profile
├── intent_detector.py        # Intent classification & RAG decision logic
├── rag_engine.py             # FAISS vector store & semantic retrieval
├── conversation_manager.py   # Session memory with windowing
├── llm_client.py             # OpenAI abstraction + stub fallback
├── evaluate.py               # Evaluation script (retrieval helped vs hurt)
├── requirements.txt
├── .env.example              # Rename to .env and add your API key
└── data/
    ├── zodiac_traits.json
    ├── planetary_impacts.json
    ├── career_guidance.txt
    ├── love_guidance.txt
    ├── spiritual_guidance.txt
    └── nakshatra_mapping.json
```

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

> ⚡ **No OpenAI key?** The agent runs in **stub mode** — rule-based responses are returned. All other features (RAG, memory, intent detection) work fully.

### 3. Run the server

```bash
python app.py
# OR
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Usage

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-001",
    "message": "What should I focus on in my career this month?",
    "user_profile": {
      "name": "Ritika",
      "birth_date": "1995-08-20",
      "birth_time": "14:30",
      "birth_place": "Jaipur, India",
      "preferred_language": "en"
    }
  }'
```

**Response:**
```json
{
  "response": "As a Leo, Jupiter's current transit is opening new doors in your career...",
  "zodiac": "Leo",
  "context_used": ["career_guidance", "zodiac_traits"],
  "retrieval_used": true,
  "intent": "career",
  "session_id": "user-001"
}
```

### Hindi language example

```json
{
  "session_id": "user-001",
  "message": "कौन सा ग्रह मेरे प्रेम जीवन को प्रभावित कर रहा है?",
  "user_profile": {
    "name": "Priya",
    "birth_date": "1998-04-10",
    "preferred_language": "hi"
  }
}
```

### Other endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Main conversational endpoint |
| `/session/{id}` | GET | Inspect session state & memory |
| `/session/{id}` | DELETE | Clear session memory |
| `/health` | GET | Health check + RAG stats |
| `/docs` | GET | Auto-generated Swagger UI |

---

## 🧠 RAG Architecture

```
User Message
    │
    ▼
Intent Detector
    ├── career / love / spiritual / planetary / zodiac / stress
    │       └──► RETRIEVE from FAISS (semantic similarity)
    │               ├── zodiac_traits.json
    │               ├── planetary_impacts.json
    │               ├── career_guidance.txt
    │               ├── love_guidance.txt
    │               └── spiritual_guidance.txt
    │
    └── summary / greeting
            └──► SKIP RETRIEVAL (use conversation memory)
    │
    ▼
Build Prompt
    ├── System: Role + User Profile + Zodiac
    ├── Memory: Last 6 turns (windowed)
    ├── Context: Retrieved chunks (if any)
    └── User message
    │
    ▼
LLM (OpenAI GPT-4o-mini / stub)
    │
    ▼
Structured Response
```

### When retrieval is skipped:
- `summary` / `recap` queries — conversation history used instead
- `greeting` / chit-chat — no domain knowledge needed  
- **Repeated intent** in last 2 turns — avoids redundant token costs

---

## 📊 Evaluation

Run the built-in evaluation suite:

```bash
python evaluate.py
```

**5 test cases covered:**

| Case | Message | Decision | Why |
|---|---|---|---|
| A | "What should I focus on in my career?" | ✅ Retrieval Used | Relevant domain docs found |
| B | "Summarize what you've told me so far" | ⛔ Bypassed | Meta-query — memory used instead |
| C | "Hello! Good morning." | ⛔ Bypassed | Greeting — no RAG needed |
| D | Hindi love query | ✅ Retrieval Used | Hindi output with domain context |
| E | Career (repeated intent) | ⛔ Bypassed | Already retrieved recently — saves tokens |

---

## 🏗️ Architecture Decisions

### Memory Control
- **Window size**: Last 10 turns kept in full detail
- **Summarization**: Turns beyond 20 get compressed into a summary string
- **Growth cap**: Prevents unbounded memory accumulation

### Cost Awareness
- Retrieval is intentionally skipped when it would add noise (meta-queries, greetings, repeated intents)
- Similarity threshold (0.30) filters low-quality retrievals
- Shorter context window (6 turns) passed to LLM to minimize token cost

### Zodiac Personalization
- Zodiac sign is appended to every retrieval query for targeted results
- Ruling planet included in every LLM prompt
- Moon sign, age, goals supported as optional stubs

---

## 🔧 Extending the Project

1. **Add more knowledge**: Drop `.txt` or `.json` files in `/data/` and register in `rag_engine.py`
2. **Persistent sessions**: Replace in-memory `ConversationManager` with Redis
3. **Better embeddings**: Swap `all-MiniLM-L6-v2` for a multilingual model for Hindi search
4. **Moon sign calculation**: Use `pyephem` or `swiss ephemeris` for precise moon sign from birth details
5. **Scoring UI**: Add a `/feedback` endpoint to collect user ratings for RAG quality
