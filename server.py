#!/usr/bin/env python3
"""
LexDraft Assist — HTTP server
Pure Python 3: http.server + socketserver. No Flask/FastAPI/Django.

Endpoints
---------
GET  /                         → index.html
GET  /api/models               → list Ollama models
GET  /api/historical           → list parsed historical JSON files
GET  /api/db_status            → ChromaDB stats
POST /api/upload_historical    → upload + parse + index a historical PDF
POST /api/upload_session_doc   → upload a temporary reference PDF (in-memory)
POST /api/analyze_grounding_text → retrieve precedents + generate draft
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

# ── Paths & constants ─────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
PARSED_DIR = BASE_DIR / "parsed_historical"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR             # index.html lives here
HOST = "127.0.0.1"
PORT = 8000

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")


# ── Markdown stripper ─────────────────────────────────────────────────────────

def strip_preamble(text: str) -> str:
    """
    If the LLM still generated a case header / background section before the
    actual finding, cut everything before the first occurrence of the expected
    finding opener so only the finding text is returned.

    Looks for the first paragraph that begins with "I have considered" or
    "I have carefully considered" (case-insensitive).  If found, returns from
    that point onward.  If not found, returns the full text unchanged.
    """
    import re
    # Find the first line starting with "I have considered" (allow leading whitespace)
    match = re.search(r"(?im)^(I have (?:carefully )?considered\b.*)", text)
    if match:
        trimmed = text[match.start():]
        if len(trimmed) > 100:          # sanity check — must be substantial
            return trimmed
    return text


def strip_markdown(text: str) -> str:
    """
    Remove common markdown artefacts from LLM output so the draft reads
    as a plain printed legal document.

    Handles: headings (#), bold/italic (**/__/_/*), code fences (```),
    blockquotes (>), unordered list markers (- / *), horizontal rules (---),
    and inline links [text](url).
    """
    import re

    # ── ATX headings: # Heading → Heading ────────────────────────────────────
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # ── Code fences ───────────────────────────────────────────────────────────
    text = re.sub(r"```[^\n]*\n?", "", text)

    # ── Blockquotes: > text → text ────────────────────────────────────────────
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    # ── Horizontal rules (---, ***, ___) → removed ───────────────────────────
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # ── Bold+italic: ***text*** or ___text___ → text ─────────────────────────
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{3}(.+?)_{3}",   r"\1", text, flags=re.DOTALL)

    # ── Bold: **text** or __text__ → text ────────────────────────────────────
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{2}(.+?)_{2}",   r"\1", text, flags=re.DOTALL)

    # ── Italic: *text* or _text_ → text ──────────────────────────────────────
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_(.+?)_",   r"\1", text, flags=re.DOTALL)

    # ── Inline links: [text](url) → text ─────────────────────────────────────
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # ── Unordered list markers at line start: "- " or "* " → nothing ─────────
    text = re.sub(r"^[\-\*]\s+", "", text, flags=re.MULTILINE)

    # ── Checkbox list markers: - [ ] or - [x] ────────────────────────────────
    text = re.sub(r"^[\-\*]\s+\[[ xX]\]\s*", "", text, flags=re.MULTILINE)

    # ── Trailing spaces left by removals ─────────────────────────────────────
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # ── Collapse 3+ consecutive blank lines → 2 ──────────────────────────────
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# ── In-memory session document store (thread-safe) ───────────────────────────

_session_docs: dict[str, dict] = {}   # { uuid: {filename, text} }
_session_lock = threading.Lock()


def _store_session_doc(filename: str, text: str) -> str:
    doc_uuid = str(uuid.uuid4())
    with _session_lock:
        _session_docs[doc_uuid] = {"filename": filename, "text": text}
    return doc_uuid


def _get_session_doc(doc_uuid: str) -> dict | None:
    with _session_lock:
        return _session_docs.get(doc_uuid)


# ── Dependency checks ─────────────────────────────────────────────────────────

def _check_dependencies():
    missing = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        missing.append("pymupdf  → pip install pymupdf")
    try:
        import chromadb
    except ImportError:
        missing.append("chromadb → pip install chromadb")
    if missing:
        print("\n[LexDraft] Missing dependencies:")
        for m in missing:
            print(f"  • {m}")
        print()
        sys.exit(1)


# ── Startup indexing ──────────────────────────────────────────────────────────

def _startup_index_unindexed():
    """
    Scan parsed_historical/ for JSON files not yet in ChromaDB and index them.
    Runs once at server start in the main thread.
    """
    import vector_db

    PARSED_DIR.mkdir(exist_ok=True)
    json_files = list(PARSED_DIR.glob("*.json"))

    if not json_files:
        logger.info("No parsed JSON files found in %s — nothing to index.", PARSED_DIR)
        return

    indexed_sources = set(vector_db.get_indexed_sources())
    to_index = [f for f in json_files if f.name not in indexed_sources]

    if not to_index:
        logger.info("All %d parsed JSON files already indexed.", len(json_files))
        return

    logger.info("Auto-indexing %d un-indexed JSON file(s)…", len(to_index))
    for jf in to_index:
        try:
            issues = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(issues, dict):
                issues = issues.get("issues", [])
            vector_db.index_issues_bulk(issues, source=jf.name)
            logger.info("  ✓ Indexed %s (%d issues)", jf.name, len(issues))
        except Exception as exc:
            logger.error("  ✗ Failed to index %s: %s", jf.name, exc)


# ── Helper utilities ──────────────────────────────────────────────────────────

def _json_response(handler: "LexDraftHandler", data: Any, status: int = 200):
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler: "LexDraftHandler", message: str, status: int = 500):
    _json_response(handler, {"error": message}, status)


def _read_body(handler: "LexDraftHandler") -> bytes:
    length = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(length)


def _save_upload(filename: str, data: bytes) -> Path:
    UPLOADS_DIR.mkdir(exist_ok=True)
    safe_name = Path(filename).name
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(data)
    return dest


# ── Threading HTTP server ─────────────────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ── Request handler ───────────────────────────────────────────────────────────

class LexDraftHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        logger.info("%s — " + fmt, self.address_string(), *args)

    # ── CORS pre-flight ────────────────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Content-Length")
        self.end_headers()

    # ── GET ────────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/api/models":
            self._api_models()
        elif path == "/api/historical":
            self._api_historical()
        elif path == "/api/db_status":
            self._api_db_status()
        else:
            # Serve static files (CSS, JS, images, etc.)
            candidate = STATIC_DIR / path.lstrip("/")
            if candidate.is_file():
                mime = self._guess_mime(candidate)
                self._serve_file(candidate, mime)
            else:
                _error(self, "Not found", 404)

    def _serve_file(self, path: Path, mime: str):
        if not path.exists():
            _error(self, f"File not found: {path.name}", 404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _guess_mime(path: Path) -> str:
        ext = path.suffix.lower()
        return {
            ".html": "text/html; charset=utf-8",
            ".css":  "text/css; charset=utf-8",
            ".js":   "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".ico":  "image/x-icon",
            ".svg":  "image/svg+xml",
        }.get(ext, "application/octet-stream")

    def _api_models(self):
        try:
            import ollama_client
            models = ollama_client.list_local_models()
            _json_response(self, {"models": models})
        except Exception as exc:
            _error(self, str(exc))

    def _api_historical(self):
        try:
            PARSED_DIR.mkdir(exist_ok=True)
            files = []
            for jf in sorted(PARSED_DIR.glob("*.json")):
                try:
                    issues = json.loads(jf.read_text(encoding="utf-8"))
                    if isinstance(issues, dict):
                        issues = issues.get("issues", [])
                    files.append(
                        {
                            "filename": jf.name,
                            "issue_count": len(issues),
                            "modified": jf.stat().st_mtime,
                        }
                    )
                except Exception:
                    files.append({"filename": jf.name, "issue_count": -1, "modified": 0})
            _json_response(self, {"documents": files})
        except Exception as exc:
            _error(self, str(exc))

    def _api_db_status(self):
        try:
            import vector_db
            _json_response(
                self,
                {
                    "total_indexed": vector_db.get_collection_count(),
                    "indexed_sources": vector_db.get_indexed_sources(),
                },
            )
        except Exception as exc:
            _error(self, str(exc))

    # ── POST ───────────────────────────────────────────────────────────────────

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/api/upload_historical":
            self._api_upload_historical()
        elif path == "/api/upload_session_doc":
            self._api_upload_session_doc()
        elif path == "/api/analyze_grounding_text":
            self._api_analyze()
        else:
            _error(self, "Not found", 404)

    # ── /api/upload_historical ─────────────────────────────────────────────────

    def _api_upload_historical(self):
        try:
            content_type = self.headers.get("Content-Type", "")
            body = _read_body(self)

            from multipart_parser import parse_multipart

            parsed = parse_multipart(body, content_type)
            files = parsed.get("files", {})
            fields = parsed.get("fields", {})

            pdf_file = files.get("file") or next(iter(files.values()), None)
            if not pdf_file:
                _error(self, "No PDF file received.", 400)
                return

            model = fields.get("model", "gemma3:4b").strip() or "gemma3:4b"
            mode = fields.get("mode", "fast").strip().lower() or "fast"
            if mode not in ("fast", "detailed"):
                mode = "fast"

            filename = pdf_file["filename"]
            pdf_path = _save_upload(filename, pdf_file["data"])

            logger.info("Processing historical PDF: %s (model=%s, mode=%s)", filename, model, mode)

            import pdf_parser
            import vector_db

            issues = pdf_parser.parse_pdf_and_extract_issues(
                str(pdf_path), model=model, mode=mode
            )

            # Persist parsed result as JSON
            PARSED_DIR.mkdir(exist_ok=True)
            json_name = Path(filename).stem + ".json"
            json_path = PARSED_DIR / json_name
            json_path.write_text(
                json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Index into ChromaDB
            ids = vector_db.index_issues_bulk(issues, source=json_name)

            _json_response(
                self,
                {
                    "status": "ok",
                    "filename": filename,
                    "issues_extracted": len(issues),
                    "indexed_count": len(ids),
                    "json_saved": json_name,
                },
            )

        except Exception as exc:
            logger.error("upload_historical error: %s", traceback.format_exc())
            _error(self, str(exc))

    # ── /api/upload_session_doc ────────────────────────────────────────────────

    def _api_upload_session_doc(self):
        try:
            content_type = self.headers.get("Content-Type", "")
            body = _read_body(self)

            from multipart_parser import parse_multipart
            import pdf_parser

            parsed = parse_multipart(body, content_type)
            files = parsed.get("files", {})
            pdf_file = files.get("file") or next(iter(files.values()), None)

            if not pdf_file:
                _error(self, "No PDF file received.", 400)
                return

            filename = pdf_file["filename"]
            pdf_path = _save_upload("session_" + filename, pdf_file["data"])

            text = pdf_parser.extract_text_from_pdf(str(pdf_path))
            doc_uuid = _store_session_doc(filename, text)

            _json_response(
                self,
                {
                    "status": "ok",
                    "uuid": doc_uuid,
                    "filename": filename,
                    "char_count": len(text),
                },
            )

        except Exception as exc:
            logger.error("upload_session_doc error: %s", traceback.format_exc())
            _error(self, str(exc))

    # ── /api/analyze_grounding_text ────────────────────────────────────────────

    def _api_analyze(self):
        try:
            body = _read_body(self)
            data = json.loads(body.decode("utf-8"))

            grounding_text: str = data.get("case_facts", "").strip()
            model: str         = data.get("model", "gemma3:4b").strip() or "gemma3:4b"
            session_uuids: list[str] = data.get("session_doc_uuids", [])
            n_results: int     = int(data.get("n_results", 5))
            num_ctx: int       = int(data.get("num_ctx", 32_768))
            # Clamp to a sane range (single ground is short; 16k min is fine)
            num_ctx = max(8_192, min(num_ctx, 131_072))

            if not grounding_text:
                _error(self, "case_facts is required.", 400)
                return

            import vector_db
            import ollama_client
            from prompts import issue_extraction_prompt, drafting_prompt
            import pdf_parser as _pdf_parser

            # ── PASS 1: Extract issue metadata from single ground ────────────
            logger.info("Pass 1 — extracting issue+section from ground (model=%s)…", model)
            extracted_meta = {}
            search_query = grounding_text[:1000]   # safe fallback
            try:
                ext_prompt = issue_extraction_prompt(grounding_text)
                ext_raw = ollama_client.chat_completion(
                    ext_prompt, model=model, timeout=60,
                    num_ctx=min(num_ctx, 16_384),   # extraction is short
                )
                extracted_meta = _pdf_parser._parse_json_from_llm(ext_raw)
                if isinstance(extracted_meta, list):
                    extracted_meta = extracted_meta[0] if extracted_meta else {}

                # Build a focused search query from the single-ground metadata
                parts = []
                if extracted_meta.get("issue"):
                    parts.append(extracted_meta["issue"])
                secs = extracted_meta.get("sections", [])
                if isinstance(secs, list) and secs:
                    parts.append("section " + " ".join(secs))
                elif isinstance(secs, str) and secs:
                    parts.append("section " + secs)
                if extracted_meta.get("ao_stand"):
                    parts.append(extracted_meta["ao_stand"])
                if extracted_meta.get("assessee_stand"):
                    parts.append(extracted_meta["assessee_stand"])
                if extracted_meta.get("search_query"):
                    parts.append(extracted_meta["search_query"])

                search_query = " | ".join(parts) if parts else grounding_text[:1000]
                logger.info(
                    "Extracted issue: %s | Sections: %s",
                    extracted_meta.get("issue"),
                    extracted_meta.get("sections"),
                )
            except Exception as ext_exc:
                logger.warning("Issue extraction failed (%s); using raw text for search.", ext_exc)

            # ── PASS 2: Semantic search with enriched query ───────────────────
            logger.info("Pass 2 — semantic search (query len=%d)…", len(search_query))
            precedents = vector_db.semantic_search_precedents(
                search_query, n_results=n_results
            )

            # Also run a secondary search on the raw grounding text and merge
            if len(grounding_text) > 200:
                secondary = vector_db.semantic_search_precedents(
                    grounding_text[:2000], n_results=max(2, n_results // 2)
                )
                # Merge — deduplicate by source+issue, keep highest score
                seen = {(p["source"], p["issue"]): p for p in precedents}
                for p in secondary:
                    key = (p["source"], p["issue"])
                    if key not in seen or p["score"] > seen[key]["score"]:
                        seen[key] = p
                precedents = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:n_results]

            logger.info("Retrieved %d precedents.", len(precedents))

            # ── Gather session document texts ─────────────────────────────────
            session_text_parts = []
            for su in session_uuids:
                doc = _get_session_doc(su)
                if doc:
                    session_text_parts.append(
                        f"[Reference: {doc['filename']}]\n{doc['text'][:6000]}"
                    )
            session_doc_text = "\n\n".join(session_text_parts)

            # ── PASS 3: Draft the findings ────────────────────────────────────
            prompt = drafting_prompt(
                grounding_text=grounding_text,
                extracted_meta=extracted_meta,
                precedents=precedents,
                session_doc_text=session_doc_text,
            )

            logger.info(
                "Pass 3 — drafting findings (model=%s, precedents=%d, session_docs=%d)…",
                model, len(precedents), len(session_uuids),
            )
            draft_raw = ollama_client.chat_completion(
                prompt, model=model, timeout=360,
                num_ctx=num_ctx,  # user-selected context window
            )
            draft = strip_markdown(strip_preamble(draft_raw))
            logger.info(
                "Draft ready | raw=%d chars → preamble-stripped+cleaned=%d chars",
                len(draft_raw), len(draft),
            )

            _json_response(
                self,
                {
                    "status": "ok",
                    "draft": draft,
                    "extracted_meta": {
                        "issues":   [extracted_meta["issue"]] if extracted_meta.get("issue") else [],
                        "sections": extracted_meta.get("sections", []) if isinstance(extracted_meta.get("sections"), list) else ([extracted_meta["sections"]] if extracted_meta.get("sections") else []),
                    },
                    "precedents_used": [
                        {
                            "source": p["source"],
                            "issue":  p["issue"],
                            "score":  p["score"],
                        }
                        for p in precedents
                    ],
                },
            )

        except json.JSONDecodeError:
            _error(self, "Invalid JSON body.", 400)
        except Exception as exc:
            logger.error("analyze error: %s", traceback.format_exc())
            _error(self, str(exc))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(
        "\n"
        "╔══════════════════════════════════════════════════╗\n"
        "║          LexDraft Assist  — starting up          ║\n"
        "╚══════════════════════════════════════════════════╝\n"
    )

    _check_dependencies()

    PARSED_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)

    logger.info("Running startup indexing check…")
    _startup_index_unindexed()

    import ollama_client
    if ollama_client.check_ollama_running():
        models = ollama_client.list_local_models()
        logger.info("Ollama reachable. Available models: %s", models or ["(none)"])
    else:
        logger.warning(
            "Ollama not reachable at %s. Make sure it is running.", ollama_client.OLLAMA_BASE_URL
        )

    server = ThreadedHTTPServer((HOST, PORT), LexDraftHandler)
    print(f"\n  Open your browser →  http://{HOST}:{PORT}\n")
    logger.info("Server listening on http://%s:%d", HOST, PORT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LexDraft] Shutting down. Goodbye.\n")
        server.server_close()


if __name__ == "__main__":
    main()
