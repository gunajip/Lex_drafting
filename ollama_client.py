"""
Ollama LLM client with automatic SDK → urllib fallback.

Primary:  ollama Python SDK   (pip install ollama)
Fallback: urllib.request       to http://localhost:11434/api/chat

Context window
--------------
Legal appellate orders are long. Grounding text alone can be 4-8k tokens,
plus precedents and reference docs. We set num_ctx = 32768 by default so
the full prompt fits without truncation. This can be overridden per-call.

Token budget at 32k context (approximate):
  System prompt overhead  ~  500
  Grounding text          ~ 6 000
  5 precedents            ~ 4 000
  Session reference doc   ~ 3 000
  Reserve for output      ~ 6 000
  ─────────────────────────────
  Total                   ~19 500  →  safe inside 32 768
"""

import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "gemma3:4b"

# Context window sent to every call. 32 768 tokens covers the full
# grounding + precedents + reference doc + output without truncation.
# For very large models with 128k support you can raise this freely.
DEFAULT_NUM_CTX = 32_768

# Sampling: low temperature keeps findings factual and authoritative.
DEFAULT_TEMPERATURE = 0.15


def _model_options(num_ctx: int) -> dict:
    return {
        "temperature": DEFAULT_TEMPERATURE,
        "num_ctx":     num_ctx,
        # Keep output focused — avoid repetitive padding
        "repeat_penalty": 1.1,
    }


# ── SDK path ───────────────────────────────────────────────────────────────────

def _chat_via_sdk(prompt: str, model: str, timeout: int, num_ctx: int) -> str:
    import ollama  # type: ignore

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=_model_options(num_ctx),
    )
    # SDK returns either a dict or a ChatResponse object
    if isinstance(response, dict):
        return response["message"]["content"]
    return response.message.content


# ── urllib fallback ────────────────────────────────────────────────────────────

def _chat_via_urllib(prompt: str, model: str, timeout: int, num_ctx: int) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = json.dumps(
        {
            "model":   model,
            "messages": [{"role": "user", "content": prompt}],
            "stream":  False,
            "options": _model_options(num_ctx),
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)
    return data["message"]["content"]


# ── Public interface ───────────────────────────────────────────────────────────

def chat_completion(
    prompt:  str,
    model:   str = DEFAULT_MODEL,
    timeout: int = 180,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """
    Send a prompt to Ollama and return the assistant's reply.

    Parameters
    ----------
    prompt  : The full prompt string.
    model   : Ollama model tag (e.g. "gemma3:4b", "llama3:8b").
    timeout : HTTP timeout in seconds.
    num_ctx : Context window size in tokens. Defaults to 32 768.
              Raise to 65 536 or 131 072 for models that support it.
    """
    token_estimate = len(prompt) // 3   # rough chars-to-tokens
    logger.info(
        "LLM call | model=%s | num_ctx=%d | prompt~%d tokens",
        model, num_ctx, token_estimate,
    )

    if token_estimate > num_ctx * 0.85:
        logger.warning(
            "Prompt (~%d tokens) is close to context limit (%d). "
            "Consider raising num_ctx or trimming inputs.",
            token_estimate, num_ctx,
        )

    # Try SDK first
    try:
        result = _chat_via_sdk(prompt, model=model, timeout=timeout, num_ctx=num_ctx)
        logger.debug("Response via SDK | len=%d chars", len(result))
        return result
    except ImportError:
        logger.info("ollama SDK not installed — using urllib fallback.")
    except Exception as sdk_exc:
        logger.warning("ollama SDK failed (%s) — trying urllib fallback.", sdk_exc)

    # urllib fallback
    result = _chat_via_urllib(prompt, model=model, timeout=timeout, num_ctx=num_ctx)
    logger.debug("Response via urllib | len=%d chars", len(result))
    return result


def list_local_models() -> list[str]:
    """Return a list of locally available Ollama model names."""
    try:
        import ollama  # type: ignore
        resp = ollama.list()
        models = resp.get("models") if isinstance(resp, dict) else getattr(resp, "models", [])
        if models:
            names = []
            for m in models:
                if isinstance(m, dict):
                    names.append(m.get("model") or m.get("name", ""))
                else:
                    names.append(getattr(m, "model", "") or getattr(m, "name", ""))
            return [n for n in names if n]
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("SDK list_models failed: %s", exc)

    try:
        url = f"{OLLAMA_BASE_URL}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", m.get("model", "")) for m in data.get("models", [])]
    except Exception as exc:
        logger.error("Cannot reach Ollama at %s: %s", OLLAMA_BASE_URL, exc)
        return []


def check_ollama_running() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False
