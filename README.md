# VectorGuide

**Semantic expert matching for any dataset — bring your own CSV.**

VectorGuide takes a plain-English query and finds the most relevant experts, coaches, therapists, mentors, or any domain specialist from your dataset — using LangChain + FAISS vector similarity search.

No hardcoded data. No API keys. Plug in your own CSV and it works.

---

## How it works

```
Your query (plain English)
        │
        ▼  HuggingFaceEmbeddings → vector
  Query embedding
        │
        ▼  FAISS cosine similarity search
  Closest expert profiles
        │
        ▼  Cold-start blending + ranking logic
  Ranked results with explanations
```

Text meaning is captured via sentence-transformer embeddings — so *"I feel overwhelmed and anxious"* matches a coach who specialises in *"stress reduction and breathwork"* even with no keyword overlap.

---

## Quick start

```bash
git clone https://github.com/your-username/vectorguide
cd vectorguide
pip install -r requirements.txt

# Try the built-in examples
python match.py --data examples/coaches.csv   --query "I want to reduce anxiety and learn meditation"
python match.py --data examples/therapists.csv --query "I am struggling with grief after losing a parent"
python match.py --data examples/mentors.csv   --query "I want to break into machine learning from software engineering"
```

---

## Bring your own data

Any CSV works. Just point VectorGuide at the right columns:

```bash
python match.py \
  --data          my_experts.csv \
  --description-field  "bio" \
  --name-field         "full_name" \
  --popularity-field   "reviews" \
  --query         "I need help with public speaking and executive presence" \
  --top-k         5
```

**Your CSV only needs two columns to work:**
- A **description column** — the rich text that gets embedded (specialties, bio, expertise)
- A **name column** — what gets shown in results

Everything else (ratings, price, location, tags) is returned as metadata automatically.

---

## Cold-start handling

New users have no history. Without a fallback, pure semantic search might surface niche experts with no track record.

VectorGuide detects new users and blends semantic score with popularity:

```
final_score = 0.70 × semantic_similarity + 0.30 × popularity_score
```

Pass `--interactions 0` for a new user, `--interactions 10` for an existing one:

```bash
# New user — popularity boost active
python match.py --data examples/coaches.csv --query "find my purpose" --interactions 0

# Returning user — pure semantic matching
python match.py --data examples/coaches.csv --query "find my purpose" --interactions 10
```

---

## Example output

```
═════════════════════════════════════════════════════════════════
  Query : "I feel burned out and want to rediscover my purpose"
═════════════════════════════════════════════════════════════════

  #1  Daniel Park
       Purpose and meaning coaching using positive psychology, values clarification...
       Strong match · 71% semantic similarity
       rating: 4.7  ·  experience_years: 10

  #2  Sofia Reyes
       Emotional intelligence and self-compassion for people recovering from burnout...
       Good match · 58% semantic similarity
       rating: 4.9  ·  experience_years: 7

  #3  Yuki Tanaka
       Spiritual development through Zen practices, journaling, and inner child work...
       Good match · 44% semantic similarity
       rating: 4.8  ·  experience_years: 6
```

---

## Project structure

```
vectorguide/
├── core/
│   └── engine.py        ← LangChain matching engine
├── config.py            ← column mapping configuration
├── match.py             ← CLI entrypoint
├── api/
│   └── main.py          ← FastAPI REST endpoint
├── examples/
│   ├── coaches.csv      ← personal development coaches
│   ├── therapists.csv   ← therapy directory
│   └── mentors.csv      ← tech mentors
└── requirements.txt
```

---

## Extend it

**Use a better embedding model:**
```python
# config.py
config = VectorGuideConfig(
    embedding_model="sentence-transformers/all-mpnet-base-v2",  # better quality
    ...
)
```

**Enable MMR diversity re-ranking** (avoid recommending near-identical experts):
```python
# core/engine.py — replace similarity_search_with_score() with:
results = self.vectorstore.max_marginal_relevance_search(query, k=top_k)
```

**Use OpenAI embeddings** (requires API key):
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```

---

## Stack

- [LangChain](https://python.langchain.com/) — document management, embedding orchestration, vector search
- [FAISS](https://faiss.ai/) — fast approximate nearest-neighbour search
- [sentence-transformers](https://sbert.net/) — `all-MiniLM-L6-v2` for local embeddings
- [FastAPI](https://fastapi.tiangolo.com/) — REST API layer
- [pandas](https://pandas.pydata.org/) — CSV loading and data handling

---

*Built as a generic tool for semantic matching problems. Works for coaches, therapists, mentors, courses, specialists — anything with a text description and a user with a goal.*
