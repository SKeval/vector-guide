"""
config.py — VectorGuide configuration
--------------------------------------
This tells the engine which columns in your CSV do what.
You never need to change the engine code — just change the config.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VectorGuideConfig:
    """
    Maps your CSV columns to VectorGuide's concepts.

    Example — coaches CSV:
        description_field  = "specialty"   ← the text that gets embedded
        name_field         = "name"         ← shown in results
        popularity_field   = "sessions_completed"  ← used for cold-start
        metadata_fields    = ["rating", "experience_years", "tags"]

    Example — therapists CSV:
        description_field  = "about"
        name_field         = "therapist_name"
        popularity_field   = "reviews_count"
        metadata_fields    = ["approach", "price_per_session", "location"]
    """

    # ── Required ──────────────────────────────────────────────────────────────

    description_field: str
    # The column whose text gets embedded and matched against the user query.
    # Should be the richest description of what this expert does.

    name_field: str
    # The column used as the display name in results.

    # ── Optional ──────────────────────────────────────────────────────────────

    id_field: Optional[str] = None
    # Unique identifier column. If None, VectorGuide auto-generates row numbers.

    popularity_field: Optional[str] = None
    # Numeric column used for cold-start blending (e.g. sessions_completed,
    # reviews_count, years_experience). If None, cold-start fallback is disabled.

    metadata_fields: list[str] = field(default_factory=list)
    # Any extra columns you want returned alongside results
    # (rating, price, location, tags, etc.)

    # ── Cold-start settings ───────────────────────────────────────────────────

    cold_start_threshold: int = 2
    # Users with fewer interactions than this get popularity blending.

    semantic_weight: float = 0.70
    # How much weight to give semantic similarity in cold-start blending.
    # popularity_weight = 1 - semantic_weight automatically.

    # ── Model settings ────────────────────────────────────────────────────────

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Any HuggingFace sentence-transformers model works here.
    # Faster alternative: "sentence-transformers/paraphrase-MiniLM-L3-v2"
    # Better quality:     "sentence-transformers/all-mpnet-base-v2"

    top_k: int = 3
    # Default number of results to return.


# ── Pre-built configs for the example datasets ────────────────────────────────
# Users can import these directly instead of building their own.

COACHES_CONFIG = VectorGuideConfig(
    description_field="specialty",
    name_field="name",
    id_field="id",
    popularity_field="sessions_completed",
    metadata_fields=["rating", "experience_years", "tags"],
)

THERAPISTS_CONFIG = VectorGuideConfig(
    description_field="about",
    name_field="therapist_name",
    id_field="id",
    popularity_field="reviews_count",
    metadata_fields=["approach", "price_per_session", "languages"],
)

MENTORS_CONFIG = VectorGuideConfig(
    description_field="expertise",
    name_field="mentor_name",
    id_field="id",
    popularity_field="mentees_helped",
    metadata_fields=["industry", "company", "years_experience"],
)
