"""Grounded outreach email drafting based strictly on resume and search sources."""

import json
from typing import Any

from llm_helper import call_llm_json

DRAFT_MAX_TOKENS = 900

_SYSTEM_PROMPT = """You are a careful, grounded outreach email writer.
Draft a formal outreach email using ONLY facts from the candidate's resume, the discovered contacts,
and the provided web sources.

Strict grounding rules:
- Use a formal, respectful tone.
- Use ONLY facts explicitly stated in the provided resume JSON, discovered contacts, and web sources.
- NEVER invent skills, projects, achievements, roles, email addresses, or facts about any organization.
- If nothing relevant about target organizations or contacts was found in the sources, explicitly say so
  (note in the skill_matches or body that specific organization details were not found, and keep the draft
  general and honest without fabricating organization facts).
- Keep the email body strictly under 200 words, including a proper formal greeting (e.g., "Dear Hiring Team,"
  or addressing the contact role/name if known from the sources) and a proper closing with the candidate's name.
- Return ONLY valid JSON with this exact schema:
{
  "subject": "",
  "body": "",
  "skill_matches": [{"skill": "", "reason": ""}],
  "source_urls": []
}
- In "source_urls", include only URLs from the provided sources that actually support statements in the draft."""


def _normalize_skill_matches(value: Any) -> list[dict[str, str]]:
    """Normalize skill_matches into a list of {skill, reason} dicts."""
    if not isinstance(value, list):
        return []
    matches: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            skill = str(item.get("skill", "") or "").strip()
            reason = str(item.get("reason", "") or "").strip()
            if skill or reason:
                matches.append({"skill": skill, "reason": reason})
    return matches


def draft_email(
    client: Any,
    resume_json: dict[str, Any],
    sources: list[dict[str, Any]],
    target: str = "",
    intent: str = "Internship",
    recipient_name: str = "",
    contacts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Draft a formal outreach email grounded in the resume, contacts, and retrieved sources."""
    allowed_urls = {
        str(src.get("url", "") or "").strip()
        for src in sources
        if str(src.get("url", "") or "").strip()
    }

    prioritized_sources = sorted(
        sources,
        key=lambda s: bool(s.get("emails")),
        reverse=True,
    )[:4]

    context: dict[str, Any] = {
        "target": (target or "").strip() or "Discovered matching organizations and contacts",
        "intent": (intent or "Internship").strip(),
        "recipient_name": (recipient_name or "").strip(),
        "resume": resume_json,
        "discovered_contacts": (contacts or [])[:6],
        "sources": [
            {
                "url": src.get("url", ""),
                "title": src.get("title", ""),
                "text": str(src.get("text", "") or "")[:350],
            }
            for src in prioritized_sources
        ],
    }
    if not sources:
        context["research_note"] = (
            "No relevant web sources were found. "
            "Do not invent facts about any organization, and state clearly that specific details were unavailable."
        )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Draft the outreach email from this verified context:\n{json.dumps(context, ensure_ascii=False)}",
        },
    ]

    raw = call_llm_json(client, messages, max_tokens=DRAFT_MAX_TOKENS, expected_type=dict)
    if not isinstance(raw, dict):
        raise ValueError("Email writer returned an unexpected JSON structure.")

    subject = str(raw.get("subject") or raw.get("email_subject") or "").strip()
    body = str(raw.get("body") or raw.get("email_body") or raw.get("content") or "").strip()

    if not body:
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "The previous JSON had an empty email body. Return valid JSON with non-empty "
                    '"subject" and "body" fields (formal outreach email under 200 words).'
                ),
            }
        ]
        raw = call_llm_json(client, retry_messages, max_tokens=DRAFT_MAX_TOKENS, expected_type=dict)
        if not isinstance(raw, dict):
            raise ValueError("Email writer returned an unexpected JSON structure.")
        subject = str(raw.get("subject") or raw.get("email_subject") or "").strip()
        body = str(raw.get("body") or raw.get("email_body") or raw.get("content") or "").strip()

    if not body:
        raise ValueError("Could not generate email body from the model response. Please try again.")

    raw_urls = raw.get("source_urls", [])
    verified_urls = [
        str(u).strip()
        for u in (raw_urls if isinstance(raw_urls, list) else [])
        if str(u).strip() in allowed_urls
    ]

    return {
        "subject": subject or f"Application for {intent or 'Opportunity'}",
        "body": body,
        "skill_matches": _normalize_skill_matches(raw.get("skill_matches")),
        "source_urls": verified_urls,
    }
