"""
Prompt templates for LexDraft Assist.

Real workflow
─────────────
The CIT(A) works ground-by-ground. For each ground they:
  1. Paste one ground heading + the assessee's submissions for that ground
  2. Receive the CIT(A) finding for that ground only
  3. Move to the next ground

Typical single-ground input (200–2 000 words):
  "Ground No. 3 — The learned AO erred in making addition of Rs.20,25,000/-
   on account of unexplained investment u/s 69...
   [AR's arguments, case law cited, facts in support...]"

Expected output (one self-contained finding block):
  Para 4.x  — "I have considered the facts of the case..."
  Para 4.x+1— AO's position as found in the order
  Para 4.x+2— Assessee's contentions summarised
  Para 4.x+3— My analysis, legal principles, engagement with cited cases
  Para 4.x+4— "Accordingly, this ground of appeal is allowed / dismissed /
               partly allowed."
"""


# ── 1. Issue / section extraction (single ground) ─────────────────────────────

def issue_extraction_prompt(grounding_text: str) -> str:
    """
    Extract the legal issue, section numbers, AO stand, assessee stand,
    and a focused semantic-search query from a single-ground input.
    """
    return f"""You are a senior legal analyst specialising in Indian Income-tax appellate law.

The text below is ONE ground of appeal plus the assessee's submissions for that ground.

Return a JSON object with exactly these keys:
  "issue"          : concise issue label, e.g. "Unexplained Investment u/s 69"
  "sections"       : list of IT Act section numbers involved, e.g. ["69", "132(4)", "143(3)"]
  "ao_stand"       : one sentence — what the AO did / what addition was made
  "assessee_stand" : one sentence — the assessee's core defence for this ground
  "search_query"   : ≤ 35-word string that best captures this dispute for semantic search
                     (include section numbers, issue type, key legal principle)

Output ONLY valid JSON — no markdown, no explanation.

--- GROUND + SUBMISSIONS ---
{grounding_text[:4000]}
--- END ---

JSON:"""


# ── 2. Batch paragraph tagging (historical PDF ingestion) ─────────────────────

def batch_tagging_prompt(paragraphs_window: list[dict]) -> str:
    """
    Classify paragraphs from a historical appellate order during PDF ingestion.
    Each element: {"para_id": int, "text": str, "is_target": bool}
    Returns JSON array: [{"para_id": int, "issue": str, "role": str}]
    """
    lines = []
    for p in paragraphs_window:
        marker = "<<<CURRENT>>> " if p["is_target"] else ""
        lines.append(f'[{p["para_id"]}] {marker}{p["text"]}')

    window_text = "\n\n".join(lines)

    return f"""You are a legal analyst specialising in Indian income-tax appellate proceedings.

Below is an extract from a CIT(A) appellate order.
Paragraphs marked <<<CURRENT>>> must be classified. Others are context only.

For each <<<CURRENT>>> paragraph return one JSON object:
  "para_id" : the integer shown in brackets
  "issue"   : concise legal issue label (e.g. "Unexplained Investment u/s 69")
  "role"    : one of FACTS | AO_POSITION | ASSESSEE_ARGUMENT | FINDING | OUTCOME | OTHER

Rules:
1. Return ONLY a valid JSON array — no prose, no markdown fences.
2. Every <<<CURRENT>>> paragraph must appear exactly once.
3. Use "OTHER" for boilerplate, dates, page headers, or procedural lines.

--- PARAGRAPHS ---
{window_text}
--- END ---

JSON array:"""


# ── 3. Fast whole-document extraction (historical PDF ingestion) ───────────────

def fast_extract_prompt(full_text: str) -> str:
    """Single-shot extraction of all legal issues from a historical order PDF."""
    return f"""You are a legal analyst specialising in Indian income-tax appellate proceedings.

Below is the text of a CIT(A) appellate order.

Extract every distinct legal issue and return a JSON array. Each element must have:
  "issue"              : concise label (e.g. "Unexplained Investment u/s 69")
  "sections"           : relevant IT Act sections as a string (e.g. "69, 132")
  "facts"              : key facts for this issue (3-5 sentences)
  "ao_position"        : AO's reasoning and the addition made
  "assessee_argument"  : assessee's defence and case law cited
  "finding"            : CIT(A)'s analysis and reasoning
  "outcome"            : final direction (e.g. "Addition upheld", "Addition deleted", "Partly allowed")

Rules:
1. Output ONLY a valid JSON array — no markdown, no explanation.
2. Empty string "" for any unavailable field.
3. Do not blend facts from different issues.

--- ORDER TEXT ---
{full_text[:25000]}
--- END ---

JSON array:"""


# ── 4. Single-ground finding prompt ───────────────────────────────────────────

def drafting_prompt(
    grounding_text: str,
    extracted_meta: dict,
    precedents: list[dict],
    session_doc_text: str = "",
) -> str:
    """
    Generate the CIT(A)'s finding for ONE specific ground of appeal.

    grounding_text  : the single ground + assessee's submissions pasted by the CIT(A)
    extracted_meta  : {"issue", "sections", "ao_stand", "assessee_stand", ...}
    precedents      : top-N semantically matched historical findings from ChromaDB
    session_doc_text: optional text from an uploaded reference PDF
    """

    # ── Format precedents (finding + outcome only — keep context tight) ────────
    prec_blocks = []
    for i, p in enumerate(precedents, 1):
        block = (
            f"PRECEDENT {i}  [{p.get('source', 'Unknown')}]\n"
            f"Issue      : {p.get('issue', '')}\n"
            f"Sections   : {p.get('sections', '')}\n"
            f"AO's stand : {p.get('ao_position', '')[:400]}\n"
            f"Defence    : {p.get('assessee_argument', '')[:400]}\n"
            f"Finding    : {p.get('finding', '')[:800]}\n"
            f"Outcome    : {p.get('outcome', '')}"
        )
        prec_blocks.append(block)
    precedents_text = (
        "\n\n".join(prec_blocks)
        if prec_blocks
        else "No matching precedents found in the database."
    )

    # ── Reference document — extract only legal facts/principles, cap size ──────
    session_section = ""
    if session_doc_text.strip():
        session_section = (
            "\n[SUPPORTING MATERIAL — reference only; do NOT copy its format or structure]\n"
            f"{session_doc_text[:6000]}\n"
            "[END SUPPORTING MATERIAL]\n"
        )

    # ── Extracted intelligence ─────────────────────────────────────────────────
    issue    = extracted_meta.get("issue", "")
    secs_raw = extracted_meta.get("sections", [])
    sections = ", ".join(secs_raw) if isinstance(secs_raw, list) else str(secs_raw)
    ao_stand  = extracted_meta.get("ao_stand", "")
    ass_stand = extracted_meta.get("assessee_stand", "")

    return f"""You are the Commissioner of Income Tax (Appeals) [CIT(A)] writing a finding.

YOUR ONLY JOB: Write the CIT(A) finding for the ONE ground pasted below.
Nothing else. No headers. No case details. No background. Just the finding.

DO NOT WRITE ANY OF THESE — they are already in the order and must not be repeated:
  - Case title, ITA number, PAN, address, appellant/respondent names
  - "IN THE OFFICE OF..." or "BEFORE THE CIT(A)..." headings
  - "Background of the case" or "Assessment Proceedings" sections
  - "Grounds of Appeal" listing
  - "AO Observations" section
  - Any summary or description of the uploaded reference document
  - "In the result, the appeal is..." (that comes after ALL grounds are decided)

YOUR OUTPUT MUST START WITH EXACTLY THIS SENTENCE (no exceptions):
"I have considered the facts of the case and the submissions made by the learned AR."

THEN follow this paragraph structure:
  Para 1 — "I have considered the facts of the case and the submissions made by the learned AR."
  Para 2 — Restate briefly what the AO did and the addition made on this ground.
  Para 3 — Summarise the assessee's contentions before me on this ground.
  Para 4+ — My analysis: examine each argument on merits, apply the relevant law,
             engage with every case law the assessee cited (apply or distinguish).
  Last   — "Accordingly, this ground of appeal is [allowed / dismissed / partly allowed]."

STRICT FORMAT RULES:
  Plain numbered paragraphs only. No markdown whatsoever.
  No # headings, no ** bold, no * bullets, no > quotes, no ``` fences.
  First person throughout: "I find...", "I am of the view...", "I direct the AO to..."

Issue           : {issue or '(see ground below)'}
IT Act Sections : {sections or '(see ground below)'}
AO's Position   : {ao_stand or '(see ground below)'}
Assessee's Stand: {ass_stand or '(see ground below)'}

GROUND + ASSESSEE'S SUBMISSIONS:
{grounding_text}
{session_section}
PRECEDENTS FOR LEGAL PRINCIPLES ONLY (never copy their facts into this case):
{precedents_text}

Begin now — first word must be "I":"""
