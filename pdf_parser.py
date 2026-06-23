"""
PDF parsing and legal-issue extraction for LexDraft Assist.

Two modes
---------
detailed : Sliding-window batch tagging (5 paragraphs per batch).
fast     : Single-shot whole-document summarisation.

Both modes call the LLM via ollama_client.chat_completion().
"""

import json
import logging
import re
import time
from typing import Any

from prompts import batch_tagging_prompt, fast_extract_prompt

logger = logging.getLogger(__name__)

MIN_PARA_LEN = 80          # Characters; shorter paragraphs are tagged OTHER
BATCH_SIZE = 5             # Target paragraphs per batch (detailed mode)
CONTEXT_WINDOW = 2         # Preceding / succeeding context paragraphs
FAST_TEXT_CAP = 25_000     # Max characters fed to LLM in fast mode


# ── LLM client (imported lazily to avoid circular imports) ────────────────────

def _llm_chat(prompt: str, model: str, timeout: int = 180, num_ctx: int = 16_384) -> str:
    """Call the Ollama LLM and return the assistant message text."""
    import ollama_client  # local module
    return ollama_client.chat_completion(prompt, model=model, timeout=timeout, num_ctx=num_ctx)


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_paragraphs_from_pdf(pdf_path: str) -> list[str]:
    """
    Extract paragraphs from a PDF using PyMuPDF (fitz).
    Collapses internal whitespace, joins multi-line paragraphs.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf") from exc

    doc = fitz.open(pdf_path)
    paragraphs: list[str] = []

    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] != 0:   # skip non-text blocks (images, etc.)
                continue
            raw_text: str = block[4]
            cleaned = re.sub(r"[ \t]+", " ", raw_text).strip()
            cleaned = re.sub(r"\n+", " ", cleaned).strip()
            if cleaned:
                paragraphs.append(cleaned)

    doc.close()
    return paragraphs


def extract_text_from_pdf(pdf_path: str) -> str:
    """Return the full text of a PDF as a single string."""
    paras = extract_paragraphs_from_pdf(pdf_path)
    return "\n\n".join(paras)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _parse_json_from_llm(raw: str) -> Any:
    """
    Attempt to parse JSON from an LLM response, handling common noise
    like markdown fences or trailing text.
    """
    raw = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to locate the outermost [...] or {...}
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        s = raw.find(start_char)
        e = raw.rfind(end_char)
        if s != -1 and e > s:
            try:
                return json.loads(raw[s : e + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Cannot parse JSON from LLM response: {raw[:300]}")


# ── Detailed (granular batch) mode ────────────────────────────────────────────

def process_detailed(pdf_path: str, model: str, progress_cb=None) -> list[dict]:
    """
    Detailed mode: classify paragraphs in batches of BATCH_SIZE with a
    CONTEXT_WINDOW sliding window.

    Returns a list of dicts:
        {"para_id": int, "text": str, "issue": str, "role": str}
    """
    paragraphs = extract_paragraphs_from_pdf(pdf_path)

    # Tag short paragraphs immediately
    tagged: list[dict] = []
    for i, p in enumerate(paragraphs):
        tagged.append(
            {
                "para_id": i,
                "text": p,
                "issue": "OTHER" if len(p) < MIN_PARA_LEN else None,
                "role": "OTHER" if len(p) < MIN_PARA_LEN else None,
            }
        )

    unclassified_ids = [t["para_id"] for t in tagged if t["issue"] is None]

    total_batches = (len(unclassified_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        "Detailed mode: %d paragraphs, %d unclassified → %d batches",
        len(paragraphs),
        len(unclassified_ids),
        total_batches,
    )

    for batch_idx in range(0, len(unclassified_ids), BATCH_SIZE):
        target_ids = unclassified_ids[batch_idx : batch_idx + BATCH_SIZE]

        # Build context window (target + CONTEXT_WINDOW before & after)
        first_target = target_ids[0]
        last_target = target_ids[-1]
        window_start = max(0, first_target - CONTEXT_WINDOW)
        window_end = min(len(tagged) - 1, last_target + CONTEXT_WINDOW)

        window = []
        for pid in range(window_start, window_end + 1):
            window.append(
                {
                    "para_id": tagged[pid]["para_id"],
                    "text": tagged[pid]["text"],
                    "is_target": tagged[pid]["para_id"] in target_ids,
                }
            )

        prompt = batch_tagging_prompt(window)

        try:
            raw = _llm_chat(prompt, model=model)
            results = _parse_json_from_llm(raw)

            if not isinstance(results, list):
                results = [results]

            for item in results:
                pid = int(item.get("para_id", -1))
                if 0 <= pid < len(tagged):
                    tagged[pid]["issue"] = item.get("issue", "OTHER")
                    tagged[pid]["role"] = item.get("role", "OTHER")

        except Exception as exc:
            logger.warning(
                "Batch %d failed (%s); marking as OTHER.", batch_idx // BATCH_SIZE, exc
            )
            for pid in target_ids:
                tagged[pid]["issue"] = tagged[pid]["issue"] or "OTHER"
                tagged[pid]["role"] = tagged[pid]["role"] or "OTHER"

        if progress_cb:
            done = min(batch_idx + BATCH_SIZE, len(unclassified_ids))
            progress_cb(done, len(unclassified_ids))

    # Fill any remaining None values
    for t in tagged:
        t["issue"] = t["issue"] or "OTHER"
        t["role"] = t["role"] or "OTHER"

    return tagged


def consolidate_by_issue(tagged: list[dict]) -> list[dict]:
    """
    Group tagged paragraphs by issue and consolidate role-based text.

    Returns list of dicts suitable for indexing:
        {issue, facts, ao_position, assessee_argument, finding, outcome}
    """
    from collections import defaultdict

    issue_map: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {
            "facts": [],
            "ao_position": [],
            "assessee_argument": [],
            "finding": [],
            "outcome": [],
        }
    )

    role_key_map = {
        "FACTS": "facts",
        "AO_POSITION": "ao_position",
        "ASSESSEE_ARGUMENT": "assessee_argument",
        "FINDING": "finding",
        "OUTCOME": "outcome",
    }

    for t in tagged:
        issue = t.get("issue", "OTHER")
        role = t.get("role", "OTHER")
        if issue == "OTHER" or role == "OTHER":
            continue
        key = role_key_map.get(role, None)
        if key:
            issue_map[issue][key].append(t["text"])

    consolidated = []
    for issue, roles in issue_map.items():
        consolidated.append(
            {
                "issue": issue,
                "facts": " ".join(roles["facts"]),
                "ao_position": " ".join(roles["ao_position"]),
                "assessee_argument": " ".join(roles["assessee_argument"]),
                "finding": " ".join(roles["finding"]),
                "outcome": " ".join(roles["outcome"]),
            }
        )
    return consolidated


# ── Fast (single-shot) mode ───────────────────────────────────────────────────

def process_fast(pdf_path: str, model: str) -> list[dict]:
    """
    Fast mode: one LLM call for the whole document.

    Returns a list of issue dicts:
        {issue, sections, facts, ao_position, assessee_argument, finding, outcome}
    """
    paragraphs = extract_paragraphs_from_pdf(pdf_path)
    filtered = [p for p in paragraphs if len(p) >= MIN_PARA_LEN]
    full_text = "\n\n".join(filtered)[:FAST_TEXT_CAP]

    prompt = fast_extract_prompt(full_text)

    # 25k chars of text ≈ 8–9k tokens; use 32k context to fit prompt + output
    raw = _llm_chat(prompt, model=model, timeout=240, num_ctx=32_768)
    issues = _parse_json_from_llm(raw)

    if not isinstance(issues, list):
        issues = [issues]

    # Normalise keys
    normalised = []
    for item in issues:
        sections_raw = item.get("sections", "")
        if isinstance(sections_raw, list):
            sections_str = ", ".join(str(s) for s in sections_raw)
        else:
            sections_str = str(sections_raw)
        normalised.append(
            {
                "issue":             item.get("issue", "Unknown"),
                "sections":          sections_str,
                "facts":             item.get("facts", ""),
                "ao_position":       item.get("ao_position", ""),
                "assessee_argument": item.get("assessee_argument", ""),
                "finding":           item.get("finding", ""),
                "outcome":           item.get("outcome", ""),
            }
        )
    return normalised


# ── Public entry point ────────────────────────────────────────────────────────

def parse_pdf_and_extract_issues(
    pdf_path: str,
    model: str,
    mode: str = "fast",
    progress_cb=None,
) -> list[dict]:
    """
    Top-level function called by the server.

    mode: "fast" (default) or "detailed"

    Returns list of issue dicts ready for ChromaDB indexing.
    """
    if mode == "detailed":
        tagged = process_detailed(pdf_path, model=model, progress_cb=progress_cb)
        return consolidate_by_issue(tagged)
    else:
        return process_fast(pdf_path, model=model)
