"""
ChromaDB vector database integration for LexDraft Assist.

Collection schema
-----------------
Each document embedded:
    text  = "Issue: {issue}\nFacts: {facts}"

Metadata stored per document:
    source          : filename of the source PDF
    issue           : issue label
    facts           : facts text
    ao_position     : AO's stance
    assessee_argument
    finding
    outcome
"""

import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "legal_precedents"

_client = None
_collection = None


def _get_collection():
    """Lazy-init: create/load the ChromaDB client and collection once."""
    global _client, _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is not installed. Run: pip install chromadb"
        ) from exc

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    _client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready. Count: %d",
        COLLECTION_NAME,
        _collection.count(),
    )
    return _collection


def _make_doc_id(source: str, issue: str, facts: str) -> str:
    """Deterministic ID so re-indexing the same data is idempotent."""
    raw = f"{source}|{issue}|{facts[:200]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def index_issue(
    source: str,
    issue: str,
    facts: str,
    ao_position: str = "",
    assessee_argument: str = "",
    finding: str = "",
    outcome: str = "",
    sections: str = "",
) -> str:
    """
    Embed and store a single legal issue in ChromaDB.

    Returns the document ID (idempotent — same data = same ID).
    """
    col = _get_collection()

    # Richer embedding: include sections for better semantic matching
    embed_text = (
        f"Issue: {issue}\n"
        f"Sections: {sections}\n"
        f"Facts: {facts}\n"
        f"AO Position: {ao_position[:500]}\n"
        f"Finding: {finding[:500]}"
    )
    doc_id = _make_doc_id(source, issue, facts)

    existing = col.get(ids=[doc_id])
    if existing["ids"]:
        logger.debug("Document %s already indexed; skipping.", doc_id)
        return doc_id

    col.add(
        documents=[embed_text],
        metadatas=[
            {
                "source":             source,
                "issue":              issue,
                "sections":           sections[:200],
                "facts":              facts[:2000],
                "ao_position":        ao_position[:1000],
                "assessee_argument":  assessee_argument[:1000],
                "finding":            finding[:2000],
                "outcome":            outcome[:500],
            }
        ],
        ids=[doc_id],
    )
    logger.info("Indexed issue '%s' from '%s' (id=%s).", issue, source, doc_id)
    return doc_id


def index_issues_bulk(issues: list[dict], source: str) -> list[str]:
    """
    Index a list of issue dicts from a single source document.

    Each dict should have keys: issue, facts, ao_position,
    assessee_argument, finding, outcome, sections (optional).
    """
    ids = []
    for item in issues:
        # Normalise sections — could be list or string
        sections_raw = item.get("sections", "")
        if isinstance(sections_raw, list):
            sections_str = ", ".join(str(s) for s in sections_raw)
        else:
            sections_str = str(sections_raw)

        doc_id = index_issue(
            source=source,
            issue=item.get("issue", "Unknown"),
            facts=item.get("facts", ""),
            ao_position=item.get("ao_position", ""),
            assessee_argument=item.get("assessee_argument", ""),
            finding=item.get("finding", ""),
            outcome=item.get("outcome", ""),
            sections=sections_str,
        )
        ids.append(doc_id)
    return ids


def semantic_search_precedents(query_text: str, n_results: int = 5) -> list[dict]:
    """
    Retrieve the top-N most similar precedents for a given query.

    Returns a list of dicts, each with keys:
        source, issue, facts, ao_position, assessee_argument, finding, outcome, score
    """
    col = _get_collection()
    if col.count() == 0:
        logger.warning("ChromaDB collection is empty; returning no results.")
        return []

    actual_n = min(n_results, col.count())
    results = col.query(
        query_texts=[query_text],
        n_results=actual_n,
        include=["metadatas", "distances"],
    )

    precedents = []
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for meta, dist in zip(metadatas, distances):
        precedents.append(
            {
                "source":            meta.get("source", ""),
                "issue":             meta.get("issue", ""),
                "sections":          meta.get("sections", ""),
                "facts":             meta.get("facts", ""),
                "ao_position":       meta.get("ao_position", ""),
                "assessee_argument": meta.get("assessee_argument", ""),
                "finding":           meta.get("finding", ""),
                "outcome":           meta.get("outcome", ""),
                "score":             round(1 - dist, 4),   # cosine similarity
            }
        )
    return precedents


def get_collection_count() -> int:
    """Return the total number of indexed documents."""
    try:
        return _get_collection().count()
    except Exception:
        return 0


def get_indexed_sources() -> list[str]:
    """Return a deduplicated list of source filenames already in ChromaDB."""
    try:
        col = _get_collection()
        if col.count() == 0:
            return []
        all_docs = col.get(include=["metadatas"])
        sources = {m.get("source", "") for m in all_docs["metadatas"]}
        return sorted(sources)
    except Exception:
        return []
