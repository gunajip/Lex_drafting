"""
Custom multipart/form-data parser.
Avoids deprecated `cgi` module; uses regex + byte splitting.
"""

import re


def parse_multipart(body: bytes, content_type: str) -> dict:
    """
    Parse a multipart/form-data body.

    Returns:
        dict with two keys:
          'fields'  -> { field_name: str_value }
          'files'   -> { field_name: {'filename': str, 'data': bytes, 'content_type': str} }
    """
    boundary = _extract_boundary(content_type)
    if not boundary:
        raise ValueError(f"Cannot extract boundary from Content-Type: {content_type}")

    boundary_bytes = boundary.encode("latin-1")

    # Split on the boundary; skip the preamble (index 0) and epilogue (last)
    delimiter = b"--" + boundary_bytes
    parts = body.split(delimiter)
    # parts[0]  = preamble (empty or whitespace)
    # parts[-1] = "--\r\n" epilogue
    parts = parts[1:-1]

    fields: dict[str, str] = {}
    files: dict[str, dict] = {}

    for part in parts:
        # Each part: \r\n<headers>\r\n\r\n<body>\r\n
        if part in (b"--", b"--\r\n", b"\r\n--"):
            continue
        # Strip leading \r\n
        if part.startswith(b"\r\n"):
            part = part[2:]
        # Strip trailing \r\n
        if part.endswith(b"\r\n"):
            part = part[:-2]

        # Split headers from body at the first double CRLF
        if b"\r\n\r\n" in part:
            header_section, body_part = part.split(b"\r\n\r\n", 1)
        else:
            continue

        headers = _parse_part_headers(header_section.decode("latin-1"))
        disposition = headers.get("content-disposition", "")
        part_content_type = headers.get("content-type", "application/octet-stream")

        field_name = _extract_disposition_param(disposition, "name")
        filename = _extract_disposition_param(disposition, "filename")

        if filename:
            files[field_name] = {
                "filename": filename,
                "data": body_part,
                "content_type": part_content_type,
            }
        elif field_name:
            try:
                fields[field_name] = body_part.decode("utf-8")
            except UnicodeDecodeError:
                fields[field_name] = body_part.decode("latin-1")

    return {"fields": fields, "files": files}


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_boundary(content_type: str) -> str | None:
    """Extract the boundary value from a Content-Type header string."""
    match = re.search(r'boundary=(?:"([^"]+)"|([^\s;]+))', content_type, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2)
    return None


def _parse_part_headers(raw: str) -> dict[str, str]:
    """Parse the MIME headers of a single part into a lowercase-key dict."""
    headers: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
    return headers


def _extract_disposition_param(disposition: str, param: str) -> str:
    """Extract a named parameter from a Content-Disposition header value."""
    pattern = rf'{param}=(?:"([^"]*)"|([\w\-\.]+))'
    match = re.search(pattern, disposition, re.IGNORECASE)
    if match:
        return match.group(1) if match.group(1) is not None else match.group(2)
    return ""
