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
import warnings
import pandas as pd
from typing import Any, Optional
from dataclasses import dataclass

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
            model_name="sentence-transformers/all-mpnet-base-v2")

        print(f"Loading embedding model: {config.embedding_model}")

        
        print("Converting data to LangChain documents...")
        documents: list[Document] = []
        for row in df.iterrows():
            document = Document(
                page_content=row[config.description_field],
                metadata={
                    "id": str(row[config.id_field]),
                    "name": row[config.name_field],
                    "popularity_score": self.popularity_map.get(str(row[config.id_field]), 0.0),
                    **{field: row[field] for field in config.metadata_fields if field in row}  # include extra metadata fields
                })
            documents.append(document)

        # ── TODO 3: Build the FAISS vectorstore ──────────────────────────────
        # Pass your documents list and self.embeddings to FAISS.from_documents().
        # Store the result as self.vectorstore.
        #
        # This one call does three things internally:
        #   1. Calls self.embeddings.embed_documents() on all page_content texts
        #   2. Stores the resulting vectors in a FAISS index
        #   3. Keeps the Documents alongside so you can retrieve them later
        #
        # Docs: https://python.langchain.com/docs/integrations/vectorstores/faiss/
        print("Building FAISS vector index...")
        # TODO 3 ↓
        raise NotImplementedError("TODO 3: build FAISS vectorstore")

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

        # ── TODO 4: Query the FAISS vectorstore ──────────────────────────────
        # Use similarity_search_with_score() to get the top candidates.
        # Fetch more than k so you have extras after filtering excluded IDs.
        # A good rule of thumb: fetch min(k * 3, total_experts).
        #
        # This returns: [(Document, float), ...]
        # The float is a DISTANCE (lower = more similar for FAISS default).
        #
        # Hint:
        #   raw = self.vectorstore.similarity_search_with_score(query, k=fetch_n)
        #
        # Docs: similarity_search_with_score section in
        # https://python.langchain.com/docs/integrations/vectorstores/faiss/
        fetch_n = min(k * 3, len(self.df))
        # TODO 4 ↓
        raise NotImplementedError("TODO 4: query the vectorstore")

        # ── TODO 5: Convert distance → similarity score ───────────────────────
        # FAISS returns L2 distance by default. You need similarity (0-1).
        # A simple conversion that works well:
        #
        #   similarity = 1 / (1 + distance)
        #
        # So distance=0 (identical) → similarity=1.0
        #    distance=1             → similarity=0.5
        #    distance=2             → similarity=0.33
        #
        # Build a list of (Document, similarity_score) tuples.
        # TODO 5 ↓
        raise NotImplementedError(
            "TODO 5: convert distance to similarity score")

        # ── TODO 6: Filter out excluded IDs ──────────────────────────────────
        # Each Document's metadata has the expert's id (you stored it in TODO 2).
        # Remove any (doc, score) pair where doc.metadata["id"] is in exclude_ids.
        # TODO 6 ↓
        raise NotImplementedError("TODO 6: filter excluded IDs")

        # ── TODO 7: Cold-start blending ───────────────────────────────────────
        # For new users (is_cold_start=True), blend semantic score with popularity.
        #
        # Formula:
        #   if is_cold_start and popularity is available:
        #       pop_score = doc.metadata.get("popularity_score", 0.0)
        #       final = config.semantic_weight * similarity + (1 - config.semantic_weight) * pop_score
        #   else:
        #       final = similarity
        #
        # Build a list of (Document, final_score, raw_similarity) tuples.
        # Keep raw_similarity separately — you'll use it in _build_reason().
        # TODO 7 ↓
        raise NotImplementedError("TODO 7: cold-start blending")

        # ── TODO 8: Sort by final score and take top_k ───────────────────────
        # Sort descending by final_score, then slice to k.
        # TODO 8 ↓
        raise NotImplementedError("TODO 8: sort and slice")

        # ── TODO 9 (stretch): MMR diversity re-ranking ────────────────────────
        # If your top results are too similar to each other (e.g. 3 mindfulness
        # coaches), use LangChain's built-in MMR search instead of TODO 4.
        #
        # Replace similarity_search_with_score() with:
        #   self.vectorstore.max_marginal_relevance_search(query, k=fetch_n, fetch_k=fetch_n*2)
        #
        # This returns Documents (no scores), so you'll need to re-score them
        # using self.embeddings.embed_query(query) + manual cosine similarity.
        # Do this only after TODO 4-8 are working.

        # ── TODO 10: Build MatchResult objects ───────────────────────────────
        # For each (doc, final_score, raw_similarity) in your top_k:
        #   - rank          : position (1-indexed)
        #   - name          : doc.metadata["name"]
        #   - description   : doc.page_content
        #   - similarity_score : raw_similarity (not the blended score)
        #   - popularity_rank  : use self._popularity_rank(doc.metadata["id"])
        #   - metadata      : pull all config.metadata_fields from doc.metadata
        #   - reason        : use self._build_reason(raw_similarity, is_cold_start)
        # TODO 10 ↓
        raise NotImplementedError("TODO 10: build MatchResult objects")

    # ── Private helpers (fully implemented — use these in your TODOs) ─────────

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
