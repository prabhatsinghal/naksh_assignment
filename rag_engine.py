"""
rag_engine.py
RAG (Retrieval-Augmented Generation) engine.

- Loads knowledge base from /data/ folder
- Embeds documents using sentence-transformers
- Stores embeddings in FAISS index
- Retrieves top-k relevant chunks given a query
- Supports similarity threshold filtering
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Will be imported lazily to avoid crash if not installed
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers or faiss not available. RAG will use keyword fallback.")


DATA_DIR = Path(__file__).parent / "data"

# Similarity threshold — chunks below this score are discarded
SIMILARITY_THRESHOLD = 0.30
TOP_K = 4


class RAGEngine:
    """
    Semantic retrieval engine backed by FAISS.
    Falls back to keyword matching if embeddings are unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.chunks: List[Dict] = []        # [{text, source, metadata}]
        self.embeddings: np.ndarray = None
        self.index = None
        self.model = None
        self._loaded = False

        if EMBEDDINGS_AVAILABLE:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)

    # ------------------------------------------------------------------
    # Knowledge Base Loading
    # ------------------------------------------------------------------

    def _load_zodiac_traits(self):
        path = DATA_DIR / "zodiac_traits.json"
        with open(path) as f:
            data = json.load(f)
        for sign, traits in data.items():
            for key, value in traits.items():
                chunk_text = f"{sign} zodiac {key}: {value}"
                self.chunks.append({
                    "text": chunk_text,
                    "source": "zodiac_traits",
                    "metadata": {"sign": sign, "field": key}
                })

    def _load_planetary_impacts(self):
        path = DATA_DIR / "planetary_impacts.json"
        with open(path) as f:
            data = json.load(f)
        for planet, info in data.items():
            if isinstance(info, dict):
                for key, value in info.items():
                    if key not in ["nature", "day"]:
                        chunk_text = f"{planet} planet {key}: {value}"
                        self.chunks.append({
                            "text": chunk_text,
                            "source": "planetary_impacts",
                            "metadata": {"planet": planet, "aspect": key}
                        })
            else:
                self.chunks.append({
                    "text": f"{planet}: {info}",
                    "source": "planetary_impacts",
                    "metadata": {"planet": planet}
                })

    def _load_text_file(self, filename: str, source_name: str):
        path = DATA_DIR / filename
        with open(path) as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
        for line in lines:
            if len(line) > 10:
                self.chunks.append({
                    "text": line,
                    "source": source_name,
                    "metadata": {}
                })

    def _load_nakshatra(self):
        path = DATA_DIR / "nakshatra_mapping.json"
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for name, info in data.items():
            text = f"Nakshatra {name}: {info.get('meaning', '')}. Traits: {info.get('traits', '')}. Ruled by {info.get('ruling_planet', '')}."
            self.chunks.append({
                "text": text,
                "source": "nakshatra_mapping",
                "metadata": {"nakshatra": name}
            })

    def load_knowledge_base(self):
        """Load all knowledge base files and build FAISS index."""
        logger.info("Loading knowledge base...")
        self._load_zodiac_traits()
        self._load_planetary_impacts()
        self._load_text_file("career_guidance.txt", "career_guidance")
        self._load_text_file("love_guidance.txt", "love_guidance")
        self._load_text_file("spiritual_guidance.txt", "spiritual_guidance")
        self._load_nakshatra()

        logger.info(f"Loaded {len(self.chunks)} chunks from knowledge base")

        if EMBEDDINGS_AVAILABLE and self.model:
            self._build_faiss_index()
        else:
            logger.warning("Running in keyword-fallback mode (no FAISS index)")

        self._loaded = True

    def _build_faiss_index(self):
        """Embed all chunks and build FAISS flat L2 index."""
        texts = [c["text"] for c in self.chunks]
        logger.info("Encoding chunks with sentence-transformer...")
        emb = self.model.encode(texts, show_progress_bar=False, batch_size=64)
        emb = np.array(emb).astype("float32")

        # Normalize for cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-10)

        self.embeddings = emb
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # Inner product = cosine after normalization
        self.index.add(emb)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors (dim={dim})")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        sources: List[str] = None,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Tuple[List[Dict], float]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: User query string
            sources: If provided, only return chunks from these source names
            top_k: Number of results to return
            threshold: Minimum similarity score to include a result

        Returns:
            (chunks_list, avg_score) tuple
        """
        if not self._loaded:
            self.load_knowledge_base()

        if EMBEDDINGS_AVAILABLE and self.index is not None:
            return self._semantic_retrieve(query, sources, top_k, threshold)
        else:
            return self._keyword_retrieve(query, sources, top_k)

    def _semantic_retrieve(self, query, sources, top_k, threshold):
        query_emb = self.model.encode([query], show_progress_bar=False)
        query_emb = np.array(query_emb).astype("float32")
        query_emb = query_emb / (np.linalg.norm(query_emb, keepdims=True) + 1e-10)

        # Retrieve more than top_k to allow filtering by source
        search_k = min(top_k * 6, len(self.chunks))
        scores, indices = self.index.search(query_emb, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if sources and chunk["source"] not in sources:
                continue
            if float(score) < threshold:
                continue
            results.append({**chunk, "score": float(score)})
            if len(results) >= top_k:
                break

        avg_score = float(np.mean([r["score"] for r in results])) if results else 0.0
        return results, avg_score

    def _keyword_retrieve(self, query, sources, top_k):
        """Simple keyword fallback when embeddings are unavailable."""
        query_words = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            if sources and chunk["source"] not in sources:
                continue
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored.append((overlap, chunk))

        scored.sort(key=lambda x: -x[0])
        results = [
            {**chunk, "score": score / max(len(query_words), 1)}
            for score, chunk in scored[:top_k]
        ]
        avg_score = float(np.mean([r["score"] for r in results])) if results else 0.0
        return results, avg_score

    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a prompt-ready context string."""
        if not chunks:
            return ""
        lines = ["--- Retrieved Astrological Context ---"]
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            score = chunk.get("score", 0)
            lines.append(f"[{i}] (Source: {source}, Score: {score:.2f}) {chunk['text']}")
        lines.append("--- End of Context ---")
        return "\n".join(lines)


# Singleton instance
_engine: RAGEngine = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.load_knowledge_base()
    return _engine
