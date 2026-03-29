"""
match.py — VectorGuide CLI
---------------------------
Run semantic matching from the command line against any CSV.

Usage examples:
  # Use a built-in example dataset
  python match.py --data examples/coaches.csv --query "I want to reduce anxiety"

  # Bring your own CSV
  python match.py --data my_data.csv \
                  --description-field "about" \
                  --name-field "full_name" \
                  --query "I need help with career transitions"

  # Return top 5 results
  python match.py --data examples/therapists.csv \
                  --query "grief and loss after divorce" \
                  --top-k 5

  # Simulate a cold-start new user
  python match.py --data examples/coaches.csv \
                  --query "I want to find my life purpose" \
                  --interactions 0

  # Simulate an existing user
  python match.py --data examples/coaches.csv \
                  --query "I want to find my life purpose" \
                  --interactions 10
"""

import argparse
import sys
from pathlib import Path

from config import VectorGuideConfig
from core.engine import VectorGuideEngine, load_csv

# ── Auto-detect config for known example datasets ─────────────────────────────
EXAMPLE_CONFIGS = {
    "coaches.csv":    {"description_field": "specialty",  "name_field": "name",          "popularity_field": "sessions_completed"},
    "therapists.csv": {"description_field": "about",      "name_field": "therapist_name", "popularity_field": "reviews_count"},
    "mentors.csv":    {"description_field": "expertise",  "name_field": "mentor_name",    "popularity_field": "mentees_helped"},
}


def build_config(args, df) -> VectorGuideConfig:
    """Build a VectorGuideConfig from CLI args, with smart defaults for example files."""
    filename = Path(args.data).name
    defaults = EXAMPLE_CONFIGS.get(filename, {})

    description_field = args.description_field or defaults.get("description_field")
    name_field        = args.name_field        or defaults.get("name_field")
    popularity_field  = args.popularity_field  or defaults.get("popularity_field")

    if not description_field:
        print(f"\n[ERROR] --description-field is required for custom CSVs.")
        print(f"        Your CSV has columns: {list(df.columns)}")
        sys.exit(1)
    if not name_field:
        print(f"\n[ERROR] --name-field is required for custom CSVs.")
        print(f"        Your CSV has columns: {list(df.columns)}")
        sys.exit(1)

    # Auto-detect metadata fields: everything that isn't a core field
    core_fields = {description_field, name_field, popularity_field, "id"}
    metadata_fields = [
        col for col in df.columns
        if col not in core_fields
        and col != description_field
    ]

    return VectorGuideConfig(
        description_field=description_field,
        name_field=name_field,
        id_field="id" if "id" in df.columns else None,
        popularity_field=popularity_field,
        metadata_fields=metadata_fields,
        top_k=args.top_k,
    )


def print_results(results, query: str, is_cold_start: bool) -> None:
    """Pretty-print match results."""
    cold_tag = "  [NEW USER — popularity boost active]" if is_cold_start else ""
    print(f"\n{'═' * 65}")
    print(f"  Query : \"{query}\"")
    print(f"  {cold_tag}")
    print(f"{'═' * 65}")

    if not results:
        print("\n  No results found.")
        return

    for r in results:
        print(f"\n  #{r.rank}  {r.name}")
        desc_preview = r.description[:80] + "..." if len(r.description) > 80 else r.description
        print(f"       {desc_preview}")
        print(f"       {r.reason}")
        if r.metadata:
            meta_str = "  ·  ".join(
                f"{k}: {v}" for k, v in r.metadata.items() if v
            )
            if meta_str:
                print(f"       {meta_str}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="VectorGuide — semantic expert matching for any CSV dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--data",              required=True,  help="Path to your CSV file")
    parser.add_argument("--query",             required=True,  help="What you are looking for in plain English")
    parser.add_argument("--description-field", default=None,   help="CSV column to embed (auto-detected for example files)")
    parser.add_argument("--name-field",        default=None,   help="CSV column for display name (auto-detected for example files)")
    parser.add_argument("--popularity-field",  default=None,   help="CSV column for popularity score (optional)")
    parser.add_argument("--top-k",             default=3, type=int, help="Number of results to return (default: 3)")
    parser.add_argument("--interactions",      default=0, type=int, help="Number of past interactions for this user (0 = new user)")
    parser.add_argument("--exclude",           default="",     help="Comma-separated IDs to exclude from results")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"\n[ERROR] File not found: {args.data}")
        sys.exit(1)

    df = load_csv(args.data)
    config = build_config(args, df)
    engine = VectorGuideEngine(df, config)

    exclude_ids = [x.strip() for x in args.exclude.split(",") if x.strip()]
    is_cold_start = args.interactions < config.cold_start_threshold

    results = engine.match(
        query=args.query,
        top_k=args.top_k,
        user_interaction_count=args.interactions,
        exclude_ids=exclude_ids,
    )

    print_results(results, args.query, is_cold_start)


if __name__ == "__main__":
    main()
