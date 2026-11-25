"""Resume text extraction (PDF/TXT) and structured JSON parsing."""

import json
from typing import Any

from pypdf import PdfReader

from llm_helper import call_llm_json

MAX_RESUME_CHARS = 6000
PARSE_MAX_TOKENS = 1000

_SYSTEM_PROMPT = """You are a precise resume parser.
Extract structured facts from the resume text and return ONLY valid JSON with these exact keys:
{
  "name": "",
  "email": "",
  "education": [],
  "skills": [],
  "projects": [{"title": "", "summary": ""}],
  "experience": []
}

Rules:
- Use ONLY facts explicitly present in the resume text.
- Do NOT invent, guess, or extrapolate any details.
- If a field is missing, use "" for strings and [] for lists.
- Keep "education" as a list of concise strings (degree, institution, dates if stated).
- Keep "experience" as a list of concise strings (role, organization, dates, key work if stated).
- Keep "projects" as a list of objects with "title" and "summary" string fields.
- Return valid JSON only, with no markdown or extra commentary."""


def _read_txt_bytes(raw: bytes) -> str:
    """Decode plain text bytes using UTF-8 with latin-1 fallback."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def extract_text(uploaded_file: Any) -> str:
    """Read text from an uploaded PDF or TXT file and truncate to 6000 characters."""
    if uploaded_file is None:
        raise ValueError("No resume file was provided.")

    filename = getattr(uploaded_file, "name", "") or ""
    lower_name = filename.lower()

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    if lower_name.endswith(".txt"):
        raw = uploaded_file.read()
        text = raw if isinstance(raw, str) else _read_txt_bytes(raw)
    elif lower_name.endswith(".pdf") or not lower_name:
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    else:
        raise ValueError("Unsupported file type. Please upload a .pdf or .txt resume.")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError(
            "Could not extract any text from the uploaded resume. "
            "If it is a scanned PDF, please upload a text-based PDF or .txt file."
        )

    return cleaned[:MAX_RESUME_CHARS]


def _as_str_list(value: Any) -> list[str]:
    """Normalize a field into a clean list of non-empty strings."""
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for entry in value:
        if isinstance(entry, dict):
            parts = [str(v).strip() for v in entry.values() if str(v).strip()]
            joined = " — ".join(parts)
            if joined:
                result.append(joined)
        else:
            item = str(entry).strip()
            if item:
                result.append(item)
    return result


def _as_projects_list(value: Any) -> list[dict[str, str]]:
    """Normalize projects into a list of {title, summary} dicts."""
    if not isinstance(value, list):
        return []
    projects: list[dict[str, str]] = []
    for entry in value:
        if isinstance(entry, dict):
            title = str(entry.get("title", "") or "").strip()
            summary = str(entry.get("summary", "") or "").strip()
            if title or summary:
                projects.append({"title": title, "summary": summary})
        elif str(entry).strip():
            projects.append({"title": str(entry).strip(), "summary": ""})
    return projects


def parse_resume(client: Any, text: str) -> dict[str, Any]:
    """Parse resume text into structured JSON without inventing facts."""
    truncated = (text or "").strip()[:MAX_RESUME_CHARS]
    if not truncated:
        raise ValueError("Resume text is empty.")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Resume text:\n\n{truncated}"},
    ]
    raw = call_llm_json(client, messages, max_tokens=PARSE_MAX_TOKENS, expected_type=dict)
    if not isinstance(raw, dict):
        raise ValueError("Resume parser returned an unexpected JSON structure.")

    return {
        "name": str(raw.get("name", "") or "").strip(),
        "email": str(raw.get("email", "") or "").strip(),
        "education": _as_str_list(raw.get("education")),
        "skills": _as_str_list(raw.get("skills")),
        "projects": _as_projects_list(raw.get("projects")),
        "experience": _as_str_list(raw.get("experience")),
    }


def format_resume_for_prompt(resume_json: dict[str, Any]) -> str:
    """Serialize structured resume JSON concisely for downstream prompts."""
    return json.dumps(resume_json, ensure_ascii=False, indent=2)
