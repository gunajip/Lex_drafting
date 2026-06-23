# LexDraft Assist

A local, privacy-first legal drafting assistant for Income Tax appellate proceedings. The system ingests historical appellate orders (PDFs), extracts and stores legal issues in a local vector database, and uses them as precedents to draft findings for new grounds of appeal — entirely on your machine, with no data sent to any cloud service.

---

## What it does

The Commissioner of Income Tax (Appeals) [CIT(A)] pastes one ground of appeal along with the assessee's submissions into the chat. The system:

1. Extracts the legal issue, relevant IT Act sections, the AO's position, and the assessee's stand from the ground text.
2. Searches the historical precedent database for semantically similar cases.
3. Generates the CIT(A) finding for that specific ground — in first person, plain numbered paragraphs, no Markdown.

An optional reference document (PDF) can be uploaded per session to supplement the finding with additional facts or legal principles.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Pure Python 3 — `http.server` + `socketserver` |
| Frontend | Vanilla HTML / CSS / JavaScript |
| LLM | Local [Ollama](https://ollama.com) models (SDK with `urllib` fallback) |
| PDF parsing | [PyMuPDF](https://pymupdf.readthedocs.io) (`fitz`) |
| Vector database | [ChromaDB](https://www.trychroma.com) (local persistent store) |

No Flask, FastAPI, Django, or cloud APIs are used.

---

## Project structure

```
Local_Legal_Drafting/
├── server.py              # Main HTTP server — all API endpoints
├── index.html             # Frontend UI (served by server.py)
├── prompts.py             # All LLM prompt templates
├── ollama_client.py       # Ollama SDK wrapper with urllib fallback
├── vector_db.py           # ChromaDB indexing and semantic search
├── pdf_parser.py          # PDF text extraction and issue extraction
├── multipart_parser.py    # Multipart form-data parser (no cgi module)
├── requirements.txt       # Python dependencies
├── setup.sh               # One-shot setup script
├── chroma_db/             # ChromaDB persistent storage (auto-created)
├── parsed_historical/     # JSON output of processed historical PDFs
└── uploads/               # Temporary uploaded files
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running locally
- A pulled Ollama model — recommended: `gemma3:4b`

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the recommended model
ollama pull gemma3:4b

# Start the Ollama server (if not already running)
ollama serve
```

### Install and run

```bash
# Clone or copy the project folder, then:
cd Local_Legal_Drafting

# Run the one-shot setup script (creates .venv and installs dependencies)
chmod +x setup.sh && ./setup.sh

# Activate the virtual environment
source .venv/bin/activate

# Start the server
python server.py
```

Open your browser at **http://127.0.0.1:8000**

### Manual install (without setup.sh)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p parsed_historical uploads chroma_db
python server.py
```

---

## Usage

### Step 1 — Ingest historical orders (one-time)

1. Click **Upload Historical PDF** in the sidebar.
2. Select one or more historical appellate order PDFs.
3. The system extracts issues, facts, AO positions, assessee arguments, and findings, then stores them in ChromaDB.
4. Each uploaded order becomes a searchable precedent for future drafting.

### Step 2 — Draft a finding for a ground

1. Paste **one complete ground of appeal and its submissions** into the input box.
   - Include the assessee's arguments, cited case law, and any relevant figures.
   - Do not paste the entire appeal document — one ground at a time.
2. Optionally, click the attachment icon to upload a **reference PDF** for the current session (e.g., a relevant order, circular, or judgment).
3. Click **Send** (or press `Enter`).
4. The system returns the CIT(A) finding for that ground — plain text, numbered paragraphs, first-person voice.

### Sidebar controls

| Control | Description |
|---|---|
| Model | Select the Ollama model to use |
| Context window | Adjust token context size (8k – 128k). Increase for very long documents |
| Upload Historical PDF | Ingest a new precedent PDF into ChromaDB |
| DB Status | Show how many issues are indexed |
| Light / Dark toggle | Switch the UI theme |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `index.html` |
| `GET` | `/api/models` | Lists available Ollama models |
| `GET` | `/api/historical` | Lists parsed historical JSON files |
| `GET` | `/api/db_status` | Returns ChromaDB collection stats |
| `POST` | `/api/upload_historical` | Uploads, parses, and indexes a historical PDF |
| `POST` | `/api/upload_session_doc` | Uploads a temporary reference document for the current session |
| `POST` | `/api/analyze_grounding_text` | Runs 3-pass analysis and returns the drafted finding |

---

## How the 3-pass analysis works

**Pass 1 — Issue extraction**
The ground text is sent to the LLM using `issue_extraction_prompt`. It returns structured metadata: the legal issue, relevant IT Act sections, the AO's position, and the assessee's stand. Context is capped at 16k tokens for speed.

**Pass 2 — Semantic search**
The extracted metadata is used to build a rich search query against ChromaDB. A primary search uses the enriched metadata; a secondary search uses the raw ground text. Results are merged and deduplicated to surface the most relevant precedents.

**Pass 3 — Drafting**
The ground text, optional reference document, and matched precedents are assembled into `drafting_prompt` and sent to the LLM. The full user-selected context window is used. The output is post-processed to strip any preamble or Markdown before being returned.

---

## Output format

The generated finding follows this paragraph structure:

```
Para X.1  I have considered the facts of the case and the submissions
          made by the learned AR on this ground.

Para X.2  The AO made an addition of Rs. ... on account of ...

Para X.3  The assessee contended that ...

Para X.4+ Analysis — each argument examined on merits, relevant law applied,
          every case law cited by the assessee addressed.

Last para Accordingly, this ground of appeal is [allowed / dismissed /
          partly allowed] for the reasons stated above.
```

Plain text only. No Markdown. First person throughout.

---

## Privacy

All processing happens locally:
- The LLM runs on your machine via Ollama.
- ChromaDB stores vectors and metadata in `chroma_db/` on your disk.
- No data is sent to any external server or API.

---

## Troubleshooting

**Server shows "Missing dependencies"**
Run `pip install -r requirements.txt` inside the activated virtual environment.

**Ollama model not found**
Run `ollama pull gemma3:4b` and ensure `ollama serve` is running before starting the server.

**Output still contains preamble or Markdown**
Increase the context window slider and retry. For very large reference documents, only the first 6000 characters are used to avoid overwhelming the model.

**ChromaDB is empty / no precedents found**
Upload at least one historical PDF via the sidebar before querying. The system can still draft a finding without precedents, but quality improves significantly with a populated database.
