"""
core/engine.py — VectorGuide Matching Engine

Before you start, read these (in order):
  1. https://python.langchain.com/docs/concepts/documents/
  2. https://python.langchain.com/docs/concepts/vectorstores/
  3. https://python.langchain.com/docs/integrations/vectorstores/faiss/
  4. https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub/

"""


from __future__ import annotations

from config import VectorGuideConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


warnings.filterwarnings("ignore")

# ── LangChain imports ─────────────────────────────────────────────────────────


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """One matched expert returned to the user."""
    rank: int
    name: str
    description: str
    similarity_score: float     # 0.0 – 1.0, higher = better match
    popularity_rank: int        # 1 = most popular in the dataset
    metadata: dict[str, Any]    # all extra fields from the CSV
    reason: str                 # human-readable explanation


# ── Engine ────────────────────────────────────────────────────────────────────

class VectorGuideEngine:
    """
    Generic semantic matching engine.

    Takes any pandas DataFrame + a VectorGuideConfig.
    Builds a FAISS vector index from the description column.
    Matches user queries to the closest rows using LangChain.

    How it all connects:

    DataFrame row (one expert)
         │
         ▼  config.description_field → text to embed
    LangChain Document(page_content=text, metadata={id, name, popularity, ...})
         │
         ▼  HuggingFaceEmbeddings converts text → vector
    FAISS vectorstore (fast nearest-neighbour search)
         │
         ▼  .similarity_search_with_score(user_query, k=N)
    [(Document, distance_score), ...]
         │
         ▼  YOUR ranking logic (cold-start blend + filter + sort)
    [MatchResult, ...]  ← returned to CLI / API
    """

    def __init__(self, df: pd.DataFrame, config: VectorGuideConfig):
        self.df = df.copy()
        self.config = config

        # Normalise popularity scores to 0-1 so they're comparable to
        # similarity scores during cold-start blending.
        # Already done for you — self.popularity_map[row_id] = 0.0 to 1.0
        self.popularity_map: dict[str, float] = {}
        if config.popularity_field and config.popularity_field in df.columns:
            max_val = df[config.popularity_field].max()
            if max_val > 0:
                for _, row in df.iterrows():
                    row_id = str(row.get(config.id_field, row.name))
                    self.popularity_map[row_id] = float(
                        row[config.popularity_field]) / max_val

        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model)

        print(f"Loading embedding model: {config.embedding_model}")

        print("Converting data to LangChain documents...")
        documents: list[Document] = []
        for _, row in df.iterrows():
            document = Document(
                page_content=row[config.description_field],
                metadata={
                    "id": str(row[config.id_field]),
                    "name": row[config.name_field],
                    "popularity_score": self.popularity_map.get(str(row[config.id_field]), 0.0),
                    # include extra metadata fields
                    **{field: row[field] for field in config.metadata_fields if field in row}
                })
            documents.append(document)

        print("Building FAISS vector index...")

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print(f"Ready — {len(documents)} experts indexed.\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def match(
        self,
        query: str,
        top_k: Optional[int] = None,
        user_interaction_count: int = 0,
        exclude_ids: Optional[list[str]] = None,
    ) -> list[MatchResult]:
        """
        Match a plain-English query against the indexed experts.

        Parameters
        ----------
        query                  : what the user is looking for
        top_k                  : how many results to return (default: config.top_k)
        user_interaction_count : sessions/interactions the user has had.
                                 Below config.cold_start_threshold → popularity blend.
        exclude_ids            : expert IDs to exclude (already seen by user)
        """
        k = top_k or self.config.top_k
        exclude_ids = set(exclude_ids or [])
        is_cold_start = user_interaction_count < self.config.cold_start_threshold

        fetch_n = min(k * 3, len(self.df))
        raw = self.vectorstore.max_marginal_relevance_search(query, k=fetch_n, fetch_k=fetch_n*2)
        
        query_vector = self.embeddings.embed_query(query)
        
        
        
        scored = []
        doc_vector = self.embeddings.embed_documents([document.page_content for document in raw])
        
        for document, doc_vector in zip(raw, doc_vector):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            scored.append((document, similarity))

        # Each Document's metadata has the expert's id).
        # Remove any (doc, score) pair where doc.metadata["id"] is in exclude_ids.
        scored = [
            (doc, sim) for doc, sim in scored
            if doc.metadata["id"] not in exclude_ids
        ]

        # For new users (is_cold_start=True), blend semantic score with popularity.

        final_scored = []
        for doc, sim in scored:
            if is_cold_start and self.popularity_map:
                pop_score = doc.metadata.get("popularity_score", 0.0)
                final_score = self.config.semantic_weight * sim + \
                    (1 - self.config.semantic_weight) * pop_score
            else:
                final_score = sim
            # keep raw similarity for reason
            final_scored.append((doc, final_score, sim))

        final_score_sorted = sorted(
            final_scored, key=lambda x: x[1], reverse=True)
        
        results = []
        for i,(doc, final_score, raw_similarity) in enumerate(final_score_sorted[:k]):

            matchresult = MatchResult(
                rank=i+1,
                name=doc.metadata["name"],
                description=doc.page_content,
                similarity_score=raw_similarity,
                popularity_rank=self._popularity_rank(doc.metadata["id"]),
                metadata={field: doc.metadata.get(
                    field) for field in self.config.metadata_fields},
                reason=self._build_reason(raw_similarity, is_cold_start)
            )
            results.append(matchresult)
        return results

    # ── Private helpers (fully implemented — use these in your TODOs) ─────────

    def _cosine_similarity(self, vec1, vec2):
            a = np.array(vec1)
            b = np.array(vec2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _popularity_rank(self, expert_id: str) -> int:
        """Returns 1-indexed popularity rank. 1 = most popular."""
        if not self.popularity_map:
            return 0
        sorted_ids = sorted(
            self.popularity_map, key=lambda k: self.popularity_map[k], reverse=True
        )
        try:
            return sorted_ids.index(expert_id) + 1
        except ValueError:
            return len(sorted_ids) + 1

    def _build_reason(self, similarity: float, is_cold_start: bool) -> str:
        """Human-readable explanation for the match."""
        if similarity > 0.55:
            strength = "Strong match"
        elif similarity > 0.40:
            strength = "Good match"
        else:
            strength = "Relevant option"

        cold_note = " · popularity-boosted (new user)" if is_cold_start else ""
        return f"{strength} · {similarity:.0%} semantic similarity{cold_note}"


# ── Loader ────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame. Strips whitespace from column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


if __name__ == "__main__":
    from config import COACHES_CONFIG
    import pandas as pd

    df = pd.read_csv("examples/coaches.csv")
    engine = VectorGuideEngine(df, COACHES_CONFIG)

    results = engine.vectorstore.similarity_search_with_score(
        "stress and anxiety", k=2)
    for doc, score in results:
        print(doc.metadata["name"])
        print(doc.page_content)
        print(f"distance score: {score}")
        print("---")
